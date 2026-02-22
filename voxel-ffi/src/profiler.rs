use std::ffi::{c_char, CString};
use std::fmt::Write as FmtWrite;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Per-phase timing statistics with min/avg/max tracking.
#[derive(Debug, Clone)]
pub struct PhaseStats {
    pub count: u64,
    pub total: Duration,
    pub min: Duration,
    pub max: Duration,
}

impl PhaseStats {
    pub fn new() -> Self {
        Self {
            count: 0,
            total: Duration::ZERO,
            min: Duration::MAX,
            max: Duration::ZERO,
        }
    }

    pub fn record(&mut self, d: Duration) {
        self.count += 1;
        self.total += d;
        if d < self.min {
            self.min = d;
        }
        if d > self.max {
            self.max = d;
        }
    }

    pub fn avg(&self) -> Duration {
        if self.count == 0 {
            Duration::ZERO
        } else {
            self.total / self.count as u32
        }
    }
}

/// Per-chunk timing breakdown recorded by worker threads.
#[derive(Debug, Clone)]
pub struct ChunkTimings {
    pub region_density: Duration,
    pub hermite: Duration,
    pub dc_solve: Duration,
    pub mesh_gen: Duration,
    pub seam_pass: Duration,
    pub coord_transform: Duration,
    pub store_read_wait: Duration,
    pub store_write_wait: Duration,
    pub total: Duration,
    pub was_slow_path: bool,
    // Mesh complexity
    pub vertex_count: u32,
    pub triangle_count: u32,
    pub section_count: u32,
    pub mesh_bytes: u32,
    // Seam sub-phase timings
    pub seam_quad_gen: Duration,
    pub seam_mesh_retrieve: Duration,
    pub seam_convert: Duration,
    pub seam_candidates_tried: u32,
    pub seam_candidates_sent: u32,
    // Send block time
    pub send_block: Duration,
    // Coarse pre-pass
    pub coarse_skip: bool,
}

/// Per-worker utilization stats.
#[derive(Debug, Clone)]
pub struct WorkerStats {
    pub total_work_time: Duration,
    pub total_idle_time: Duration,
    pub chunks_processed: u64,
    pub stale_skipped: u64,
    pub total_seam_time: Duration,
    pub total_send_block_time: Duration,
}

impl WorkerStats {
    pub fn new() -> Self {
        Self {
            total_work_time: Duration::ZERO,
            total_idle_time: Duration::ZERO,
            chunks_processed: 0,
            stale_skipped: 0,
            total_seam_time: Duration::ZERO,
            total_send_block_time: Duration::ZERO,
        }
    }
}

/// Per-chunk detail for the "top N slowest" report and per-chunk dump.
#[derive(Debug, Clone)]
pub struct ChunkDetail {
    pub coord: (i32, i32, i32),
    pub was_slow_path: bool,
    pub coarse_skip: bool,
    pub total: Duration,
    pub vertex_count: u32,
    pub triangle_count: u32,
    pub section_count: u32,
    pub mesh_bytes: u32,
    /// Per-submesh breakdown: (material_id, vertex_count, index_count)
    pub submesh_info: Vec<(u8, u32, u32)>,
}

/// Aggregated session metrics. Protected by Mutex inside StreamingProfiler.
pub struct SessionMetrics {
    pub session_id: u64,
    pub session_start: Option<Instant>,
    pub session_end: Option<Instant>,

    // Per-phase aggregated stats
    pub region_density: PhaseStats,
    pub hermite: PhaseStats,
    pub dc_solve: PhaseStats,
    pub mesh_gen: PhaseStats,
    pub seam_pass: PhaseStats,
    pub coord_transform: PhaseStats,
    pub store_read_wait: PhaseStats,
    pub store_write_wait: PhaseStats,
    pub total_chunk: PhaseStats,

    // Queue depth samples
    pub gen_queue_depths: Vec<u32>,
    pub result_queue_depths: Vec<u32>,

    // Worker stats
    pub worker_stats: Vec<WorkerStats>,

    // Chunk details for breakdown
    pub chunk_details: Vec<ChunkDetail>,

    // Seam sub-phase stats
    pub seam_quad_gen: PhaseStats,
    pub seam_mesh_retrieve: PhaseStats,
    pub seam_convert: PhaseStats,
    pub send_block: PhaseStats,
    pub coarse_skip_count: u64,
    pub coarse_full_gen_count: u64,

    // Path counts
    pub slow_path_count: u64,
    pub fast_path_count: u64,

    // Request-to-enqueue timestamps (for latency tracking)
    pub request_timestamps: Vec<((i32, i32, i32), Instant)>,
    pub request_to_result: Vec<Duration>,

    // Error count
    pub error_count: u64,

    // Config snapshot
    pub config_snapshot: String,

    // Stall frame tracking (frames where no results were ready)
    pub stall_frames: u64,
    pub total_poll_frames: u64,

    // Mesh complexity aggregated
    pub mesh_vertex_counts: Vec<u32>,
    pub mesh_triangle_counts: Vec<u32>,
    pub mesh_section_counts: Vec<u32>,
    pub mesh_bytes: Vec<u32>,
}

impl SessionMetrics {
    fn new(num_workers: usize) -> Self {
        Self {
            session_id: 0,
            session_start: None,
            session_end: None,
            region_density: PhaseStats::new(),
            hermite: PhaseStats::new(),
            dc_solve: PhaseStats::new(),
            mesh_gen: PhaseStats::new(),
            seam_pass: PhaseStats::new(),
            coord_transform: PhaseStats::new(),
            store_read_wait: PhaseStats::new(),
            store_write_wait: PhaseStats::new(),
            total_chunk: PhaseStats::new(),
            seam_quad_gen: PhaseStats::new(),
            seam_mesh_retrieve: PhaseStats::new(),
            seam_convert: PhaseStats::new(),
            send_block: PhaseStats::new(),
            coarse_skip_count: 0,
            coarse_full_gen_count: 0,
            gen_queue_depths: Vec::new(),
            result_queue_depths: Vec::new(),
            worker_stats: (0..num_workers).map(|_| WorkerStats::new()).collect(),
            chunk_details: Vec::new(),
            slow_path_count: 0,
            fast_path_count: 0,
            request_timestamps: Vec::new(),
            request_to_result: Vec::new(),
            error_count: 0,
            config_snapshot: String::new(),
            stall_frames: 0,
            total_poll_frames: 0,
            mesh_vertex_counts: Vec::new(),
            mesh_triangle_counts: Vec::new(),
            mesh_section_counts: Vec::new(),
            mesh_bytes: Vec::new(),
        }
    }

    fn reset(&mut self, num_workers: usize) {
        self.session_start = None;
        self.session_end = None;
        self.region_density = PhaseStats::new();
        self.hermite = PhaseStats::new();
        self.dc_solve = PhaseStats::new();
        self.mesh_gen = PhaseStats::new();
        self.seam_pass = PhaseStats::new();
        self.coord_transform = PhaseStats::new();
        self.store_read_wait = PhaseStats::new();
        self.store_write_wait = PhaseStats::new();
        self.total_chunk = PhaseStats::new();
        self.seam_quad_gen = PhaseStats::new();
        self.seam_mesh_retrieve = PhaseStats::new();
        self.seam_convert = PhaseStats::new();
        self.send_block = PhaseStats::new();
        self.coarse_skip_count = 0;
        self.coarse_full_gen_count = 0;
        self.gen_queue_depths.clear();
        self.result_queue_depths.clear();
        self.worker_stats = (0..num_workers).map(|_| WorkerStats::new()).collect();
        self.chunk_details.clear();
        self.slow_path_count = 0;
        self.fast_path_count = 0;
        self.request_timestamps.clear();
        self.request_to_result.clear();
        self.error_count = 0;
        self.config_snapshot.clear();
        self.stall_frames = 0;
        self.total_poll_frames = 0;
        self.mesh_vertex_counts.clear();
        self.mesh_triangle_counts.clear();
        self.mesh_section_counts.clear();
        self.mesh_bytes.clear();
    }
}

/// Thread-safe streaming profiler. Zero overhead when disabled.
pub struct StreamingProfiler {
    enabled: AtomicBool,
    session_counter: AtomicU64,
    metrics: Mutex<SessionMetrics>,
    num_workers: usize,
}

impl StreamingProfiler {
    pub fn new(num_workers: usize) -> Self {
        Self {
            enabled: AtomicBool::new(false),
            session_counter: AtomicU64::new(0),
            metrics: Mutex::new(SessionMetrics::new(num_workers)),
            num_workers,
        }
    }

    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::Relaxed);
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }

    /// Begin a new profiling session. Resets all metrics and stores the config snapshot.
    /// Returns the session id.
    pub fn begin_session(&self, config_snapshot: String) -> u64 {
        let id = self.session_counter.fetch_add(1, Ordering::Relaxed) + 1;
        let mut m = self.metrics.lock().unwrap();
        m.reset(self.num_workers);
        m.session_id = id;
        m.session_start = Some(Instant::now());
        m.config_snapshot = config_snapshot;
        id
    }

    /// Record a chunk generation request with its timestamp and coordinates.
    pub fn record_request(&self, chunk: (i32, i32, i32)) {
        if !self.is_enabled() {
            return;
        }
        let mut m = self.metrics.lock().unwrap();
        m.request_timestamps.push((chunk, Instant::now()));
    }

    /// Record that a worker skipped a stale chunk.
    pub fn record_stale_skip(&self, worker_id: usize) {
        if !self.is_enabled() {
            return;
        }
        let mut m = self.metrics.lock().unwrap();
        if let Some(ws) = m.worker_stats.get_mut(worker_id) {
            ws.stale_skipped += 1;
        }
    }

    /// Record completed chunk timings from a worker.
    pub fn record_chunk(
        &self,
        worker_id: usize,
        timings: ChunkTimings,
        gen_queue_len: u32,
        result_queue_len: u32,
    ) {
        if !self.is_enabled() {
            return;
        }
        let mut m = self.metrics.lock().unwrap();

        // Phase stats
        m.region_density.record(timings.region_density);
        m.hermite.record(timings.hermite);
        m.dc_solve.record(timings.dc_solve);
        m.mesh_gen.record(timings.mesh_gen);
        m.seam_pass.record(timings.seam_pass);
        m.coord_transform.record(timings.coord_transform);
        m.store_read_wait.record(timings.store_read_wait);
        m.store_write_wait.record(timings.store_write_wait);
        m.total_chunk.record(timings.total);

        // Seam sub-phase stats
        m.seam_quad_gen.record(timings.seam_quad_gen);
        m.seam_mesh_retrieve.record(timings.seam_mesh_retrieve);
        m.seam_convert.record(timings.seam_convert);
        m.send_block.record(timings.send_block);
        if timings.coarse_skip {
            m.coarse_skip_count += 1;
        } else {
            m.coarse_full_gen_count += 1;
        }

        // Queue depth samples
        m.gen_queue_depths.push(gen_queue_len);
        m.result_queue_depths.push(result_queue_len);

        // Worker stats
        if let Some(ws) = m.worker_stats.get_mut(worker_id) {
            ws.total_work_time += timings.total;
            ws.chunks_processed += 1;
            ws.total_seam_time += timings.seam_pass;
            ws.total_send_block_time += timings.send_block;
        }

        // Path tracking
        if timings.was_slow_path {
            m.slow_path_count += 1;
        } else {
            m.fast_path_count += 1;
        }

        // Mesh complexity
        m.mesh_vertex_counts.push(timings.vertex_count);
        m.mesh_triangle_counts.push(timings.triangle_count);
        m.mesh_section_counts.push(timings.section_count);
        m.mesh_bytes.push(timings.mesh_bytes);

        // Chunk detail
        m.chunk_details.push(ChunkDetail {
            coord: (0, 0, 0), // Will be set by caller or via record_chunk_with_coord
            was_slow_path: timings.was_slow_path,
            coarse_skip: timings.coarse_skip,
            total: timings.total,
            vertex_count: timings.vertex_count,
            triangle_count: timings.triangle_count,
            section_count: timings.section_count,
            mesh_bytes: timings.mesh_bytes,
            submesh_info: Vec::new(),
        });
    }

    /// Record completed chunk timings with chunk coordinate.
    pub fn record_chunk_with_coord(
        &self,
        worker_id: usize,
        chunk: (i32, i32, i32),
        timings: ChunkTimings,
        gen_queue_len: u32,
        result_queue_len: u32,
    ) {
        if !self.is_enabled() {
            return;
        }
        let now = Instant::now();
        let mut m = self.metrics.lock().unwrap();

        // Phase stats
        m.region_density.record(timings.region_density);
        m.hermite.record(timings.hermite);
        m.dc_solve.record(timings.dc_solve);
        m.mesh_gen.record(timings.mesh_gen);
        m.seam_pass.record(timings.seam_pass);
        m.coord_transform.record(timings.coord_transform);
        m.store_read_wait.record(timings.store_read_wait);
        m.store_write_wait.record(timings.store_write_wait);
        m.total_chunk.record(timings.total);

        // Seam sub-phase stats
        m.seam_quad_gen.record(timings.seam_quad_gen);
        m.seam_mesh_retrieve.record(timings.seam_mesh_retrieve);
        m.seam_convert.record(timings.seam_convert);
        m.send_block.record(timings.send_block);
        if timings.coarse_skip {
            m.coarse_skip_count += 1;
        } else {
            m.coarse_full_gen_count += 1;
        }

        // Queue depth samples
        m.gen_queue_depths.push(gen_queue_len);
        m.result_queue_depths.push(result_queue_len);

        // Worker stats
        if let Some(ws) = m.worker_stats.get_mut(worker_id) {
            ws.total_work_time += timings.total;
            ws.chunks_processed += 1;
            ws.total_seam_time += timings.seam_pass;
            ws.total_send_block_time += timings.send_block;
        }

        // Path tracking
        if timings.was_slow_path {
            m.slow_path_count += 1;
        } else {
            m.fast_path_count += 1;
        }

        // Mesh complexity
        m.mesh_vertex_counts.push(timings.vertex_count);
        m.mesh_triangle_counts.push(timings.triangle_count);
        m.mesh_section_counts.push(timings.section_count);
        m.mesh_bytes.push(timings.mesh_bytes);

        // Request-to-result latency: find matching request timestamp
        if let Some(pos) = m.request_timestamps.iter().position(|(c, _)| *c == chunk) {
            let (_, req_time) = m.request_timestamps.remove(pos);
            m.request_to_result.push(now - req_time);
        }

        // Chunk detail
        m.chunk_details.push(ChunkDetail {
            coord: chunk,
            was_slow_path: timings.was_slow_path,
            coarse_skip: timings.coarse_skip,
            total: timings.total,
            vertex_count: timings.vertex_count,
            triangle_count: timings.triangle_count,
            section_count: timings.section_count,
            mesh_bytes: timings.mesh_bytes,
            submesh_info: Vec::new(),
        });
    }

    /// Attach submesh info to the most recently recorded chunk detail.
    pub fn attach_submesh_info(&self, info: Vec<(u8, u32, u32)>) {
        if !self.is_enabled() {
            return;
        }
        let mut m = self.metrics.lock().unwrap();
        if let Some(last) = m.chunk_details.last_mut() {
            last.submesh_info = info;
        }
    }

    /// Record worker idle time (time spent waiting for work).
    pub fn record_worker_idle(&self, worker_id: usize, duration: Duration) {
        if !self.is_enabled() {
            return;
        }
        let mut m = self.metrics.lock().unwrap();
        if let Some(ws) = m.worker_stats.get_mut(worker_id) {
            ws.total_idle_time += duration;
        }
    }

    /// Record a poll frame (for stall tracking).
    pub fn record_poll(&self, had_result: bool) {
        if !self.is_enabled() {
            return;
        }
        let mut m = self.metrics.lock().unwrap();
        m.total_poll_frames += 1;
        if !had_result {
            m.stall_frames += 1;
        }
    }

    /// Record an error.
    pub fn record_error(&self) {
        if !self.is_enabled() {
            return;
        }
        let mut m = self.metrics.lock().unwrap();
        m.error_count += 1;
    }

    /// End the current profiling session.
    pub fn end_session(&self) {
        let mut m = self.metrics.lock().unwrap();
        m.session_end = Some(Instant::now());
    }

    /// Generate a comprehensive plain-text profiling report.
    pub fn generate_report(&self) -> String {
        let m = self.metrics.lock().unwrap();
        let mut out = String::with_capacity(4096);

        let wall_time = match (m.session_start, m.session_end) {
            (Some(s), Some(e)) => e.duration_since(s),
            (Some(s), None) => Instant::now().duration_since(s),
            _ => Duration::ZERO,
        };

        // ── 1. Session Header ──
        let _ = writeln!(out, "================================================================");
        let _ = writeln!(out, "  STREAMING PROFILER REPORT  -  Session #{}", m.session_id);
        let _ = writeln!(out, "================================================================");
        let _ = writeln!(out, "  Wall time:    {:.3}s", wall_time.as_secs_f64());
        let _ = writeln!(out, "  Workers:      {}", self.num_workers);
        let _ = writeln!(out);

        // ── 2. Request Summary ──
        let total_requested = m.total_chunk.count + m.error_count
            + m.worker_stats.iter().map(|w| w.stale_skipped).sum::<u64>();
        let _ = writeln!(out, "── Request Summary ──────────────────────────────────────────");
        let _ = writeln!(out, "  Requested:    {}", total_requested);
        let _ = writeln!(out, "  Completed:    {}", m.total_chunk.count);
        let _ = writeln!(out, "  Stale skip:   {}", m.worker_stats.iter().map(|w| w.stale_skipped).sum::<u64>());
        let _ = writeln!(out, "  Errors:       {}", m.error_count);
        let _ = writeln!(out, "  Slow path:    {}", m.slow_path_count);
        let _ = writeln!(out, "  Fast path:    {}", m.fast_path_count);
        let _ = writeln!(out);

        // ── 3. Per-Phase Timing Table ──
        let _ = writeln!(out, "── Per-Phase Timing (ms) ───────────────────────────────────");
        let _ = writeln!(out, "  {:<20} {:>8} {:>8} {:>8} {:>10}", "Phase", "Min", "Avg", "Max", "Total");
        let _ = writeln!(out, "  {:-<20} {:->8} {:->8} {:->8} {:->10}", "", "", "", "", "");

        let phases: &[(&str, &PhaseStats)] = &[
            ("region_density",  &m.region_density),
            ("hermite",         &m.hermite),
            ("dc_solve",        &m.dc_solve),
            ("mesh_gen",        &m.mesh_gen),
            ("seam_pass",       &m.seam_pass),
            ("coord_transform", &m.coord_transform),
            ("store_read_wait", &m.store_read_wait),
            ("store_write_wait",&m.store_write_wait),
            ("TOTAL",           &m.total_chunk),
        ];
        for (name, ps) in phases {
            if ps.count == 0 {
                let _ = writeln!(out, "  {:<20} {:>8} {:>8} {:>8} {:>10}", name, "-", "-", "-", "-");
            } else {
                let _ = writeln!(
                    out,
                    "  {:<20} {:>8.2} {:>8.2} {:>8.2} {:>10.1}",
                    name,
                    dur_ms(ps.min),
                    dur_ms(ps.avg()),
                    dur_ms(ps.max),
                    dur_ms(ps.total),
                );
            }
        }
        let _ = writeln!(out);

        // ── 3b. Seam Pass Breakdown ──
        let _ = writeln!(out, "── Seam Pass Breakdown (ms) ────────────────────────────────");
        let _ = writeln!(out, "  {:<20} {:>8} {:>8} {:>8} {:>10}", "Phase", "Min", "Avg", "Max", "Total");
        let _ = writeln!(out, "  {:-<20} {:->8} {:->8} {:->8} {:->10}", "", "", "", "", "");
        let seam_phases: &[(&str, &PhaseStats)] = &[
            ("quad_gen",      &m.seam_quad_gen),
            ("mesh_retrieve", &m.seam_mesh_retrieve),
            ("convert",       &m.seam_convert),
        ];
        for (name, ps) in seam_phases {
            if ps.count == 0 {
                let _ = writeln!(out, "  {:<20} {:>8} {:>8} {:>8} {:>10}", name, "-", "-", "-", "-");
            } else {
                let _ = writeln!(
                    out,
                    "  {:<20} {:>8.2} {:>8.2} {:>8.2} {:>10.1}",
                    name,
                    dur_ms(ps.min),
                    dur_ms(ps.avg()),
                    dur_ms(ps.max),
                    dur_ms(ps.total),
                );
            }
        }
        let total_tried: u64 = m.seam_quad_gen.count;
        let total_sent: u64 = m.seam_convert.count;
        let hit_rate = if total_tried > 0 { (total_sent as f64 / total_tried as f64) * 100.0 } else { 0.0 };
        let _ = writeln!(out, "  Candidates:   tried={}  sent={}  (hit rate: {:.1}%)", total_tried, total_sent, hit_rate);
        let _ = writeln!(out);

        // ── 3c. Send Block ──
        let _ = writeln!(out, "── Send Block (ms) ─────────────────────────────────────────");
        if m.send_block.count == 0 {
            let _ = writeln!(out, "  (no data)");
        } else {
            let _ = writeln!(out, "  Min:    {:.2}  Avg:  {:.2}  Max:  {:.2}  Total: {:.1}",
                dur_ms(m.send_block.min),
                dur_ms(m.send_block.avg()),
                dur_ms(m.send_block.max),
                dur_ms(m.send_block.total));
        }
        let _ = writeln!(out);

        // ── 3d. Coarse Pre-Pass ──
        let _ = writeln!(out, "── Coarse Pre-Pass ─────────────────────────────────────────");
        let coarse_total = m.coarse_skip_count + m.coarse_full_gen_count;
        let skip_rate = if coarse_total > 0 { (m.coarse_skip_count as f64 / coarse_total as f64) * 100.0 } else { 0.0 };
        let _ = writeln!(out, "  Skipped:  {}  Full gen: {}  (skip rate: {:.1}%)",
            m.coarse_skip_count, m.coarse_full_gen_count, skip_rate);
        let _ = writeln!(out);

        // ── 4. Lock Contention Stats ──
        let _ = writeln!(out, "── Lock Contention (ms) ────────────────────────────────────");
        let _ = writeln!(out, "  Store read wait:   avg={:.2}  max={:.2}  total={:.1}",
            dur_ms(m.store_read_wait.avg()),
            dur_ms(m.store_read_wait.max),
            dur_ms(m.store_read_wait.total));
        let _ = writeln!(out, "  Store write wait:  avg={:.2}  max={:.2}  total={:.1}",
            dur_ms(m.store_write_wait.avg()),
            dur_ms(m.store_write_wait.max),
            dur_ms(m.store_write_wait.total));
        let _ = writeln!(out);

        // ── 5. Queue Depth Stats ──
        let _ = writeln!(out, "── Queue Depth ─────────────────────────────────────────────");
        if m.gen_queue_depths.is_empty() {
            let _ = writeln!(out, "  (no samples)");
        } else {
            let (gmin, gavg, gmax) = vec_stats(&m.gen_queue_depths);
            let (rmin, ravg, rmax) = vec_stats(&m.result_queue_depths);
            let _ = writeln!(out, "  Gen queue:    min={}  avg={:.1}  max={}", gmin, gavg, gmax);
            let _ = writeln!(out, "  Result queue: min={}  avg={:.1}  max={}", rmin, ravg, rmax);
        }
        let _ = writeln!(out);

        // ── 6. Worker Utilization Table ──
        let _ = writeln!(out, "── Worker Utilization ──────────────────────────────────────");
        let _ = writeln!(out, "  {:<8} {:>8} {:>10} {:>10} {:>10} {:>10} {:>8} {:>8}",
            "Worker", "Chunks", "Work(ms)", "Idle(ms)", "Seam(ms)", "Send(ms)", "Stale", "Util%");
        let _ = writeln!(out, "  {:-<8} {:->8} {:->10} {:->10} {:->10} {:->10} {:->8} {:->8}", "", "", "", "", "", "", "", "");
        for (i, ws) in m.worker_stats.iter().enumerate() {
            let total_time = ws.total_work_time + ws.total_idle_time;
            let util_pct = if total_time.as_nanos() > 0 {
                (ws.total_work_time.as_secs_f64() / total_time.as_secs_f64()) * 100.0
            } else if ws.chunks_processed > 0 {
                100.0
            } else {
                0.0
            };
            let _ = writeln!(
                out,
                "  {:<8} {:>8} {:>10.1} {:>10.1} {:>10.1} {:>10.1} {:>8} {:>7.1}",
                i,
                ws.chunks_processed,
                dur_ms(ws.total_work_time),
                dur_ms(ws.total_idle_time),
                dur_ms(ws.total_seam_time),
                dur_ms(ws.total_send_block_time),
                ws.stale_skipped,
                util_pct,
            );
        }
        let _ = writeln!(out);

        // ── 7. Mesh Complexity Stats ──
        let _ = writeln!(out, "── Mesh Complexity ─────────────────────────────────────────");
        if m.mesh_vertex_counts.is_empty() {
            let _ = writeln!(out, "  (no data)");
        } else {
            let (vmin, vavg, vmax) = vec_stats(&m.mesh_vertex_counts);
            let (tmin, tavg, tmax) = vec_stats(&m.mesh_triangle_counts);
            let (smin, savg, smax) = vec_stats(&m.mesh_section_counts);
            let total_bytes: u64 = m.mesh_bytes.iter().map(|&b| b as u64).sum();
            let _ = writeln!(out, "  Vertices:   min={}  avg={:.0}  max={}", vmin, vavg, vmax);
            let _ = writeln!(out, "  Triangles:  min={}  avg={:.0}  max={}", tmin, tavg, tmax);
            let _ = writeln!(out, "  Sections:   min={}  avg={:.0}  max={}", smin, savg, smax);
            let _ = writeln!(out, "  Total mesh: {:.1} KB", total_bytes as f64 / 1024.0);
        }
        let _ = writeln!(out);

        // ── 8. Top 10 Slowest Chunks ──
        let _ = writeln!(out, "── Top 10 Slowest Chunks ───────────────────────────────────");
        if m.chunk_details.is_empty() {
            let _ = writeln!(out, "  (no data)");
        } else {
            let mut sorted: Vec<_> = m.chunk_details.iter().collect();
            sorted.sort_by(|a, b| b.total.cmp(&a.total));
            let _ = writeln!(out, "  {:<18} {:>8} {:>6} {:>8} {:>8}",
                "Coord", "Time(ms)", "Path", "Verts", "Tris");
            let _ = writeln!(out, "  {:-<18} {:->8} {:->6} {:->8} {:->8}", "", "", "", "", "");
            for cd in sorted.iter().take(10) {
                let path = if cd.was_slow_path { "slow" } else { "fast" };
                let _ = writeln!(
                    out,
                    "  ({:>4},{:>4},{:>4}) {:>8.2} {:>6} {:>8} {:>8}",
                    cd.coord.0, cd.coord.1, cd.coord.2,
                    dur_ms(cd.total),
                    path,
                    cd.vertex_count,
                    cd.triangle_count,
                );
            }
        }
        let _ = writeln!(out);

        // ── 8b. Per-Chunk Detail (non-empty only, UE coords) ──
        let _ = writeln!(out, "── Per-Chunk Detail (non-empty, UE coords) ─────────────────");
        let _ = writeln!(out, "  UE Coord = Rust (x, -z, y).  Submeshes: mat:V/I");
        {
            let mut non_empty: Vec<_> = m.chunk_details.iter()
                .filter(|cd| cd.vertex_count > 0)
                .collect();
            non_empty.sort_by_key(|cd| (cd.coord.0, cd.coord.1, cd.coord.2));

            if non_empty.is_empty() {
                let _ = writeln!(out, "  (no non-empty chunks)");
            } else {
                let _ = writeln!(out, "  {:<16} {:>6} {:>6} {:>4} {:>7} {:>5} {}",
                    "UE Coord", "Verts", "Tris", "Secs", "Bytes", "Path", "Submeshes");
                let _ = writeln!(out, "  {:-<16} {:->6} {:->6} {:->4} {:->7} {:->5} {:->30}",
                    "", "", "", "", "", "", "");
                for cd in &non_empty {
                    // Convert Rust coords to UE: (x, -z, y)
                    let ue = (cd.coord.0, -cd.coord.2, cd.coord.1);
                    let path = if cd.was_slow_path { "slow" } else { "fast" };
                    let mut subs = String::new();
                    for (i, &(mat, vc, ic)) in cd.submesh_info.iter().enumerate() {
                        if i > 0 { subs.push_str(", "); }
                        let _ = write!(subs, "{}:{}/{}", mat, vc, ic);
                    }
                    if subs.is_empty() { subs.push_str("-"); }
                    let _ = writeln!(out, "  ({:>4},{:>4},{:>4}) {:>6} {:>6} {:>4} {:>7} {:>5} [{}]",
                        ue.0, ue.1, ue.2,
                        cd.vertex_count, cd.triangle_count, cd.section_count,
                        cd.mesh_bytes, path, subs);
                }
                let _ = writeln!(out, "  Total non-empty: {}", non_empty.len());
            }
        }
        let _ = writeln!(out);

        // ── 9. Generation Config Snapshot ──
        let _ = writeln!(out, "── Generation Config ───────────────────────────────────────");
        if m.config_snapshot.is_empty() {
            let _ = writeln!(out, "  (none captured)");
        } else {
            for line in m.config_snapshot.lines() {
                let _ = writeln!(out, "  {}", line);
            }
        }
        let _ = writeln!(out);

        // ── 10. Request-to-Result Latency ──
        let _ = writeln!(out, "── Request-to-Result Latency (ms) ──────────────────────────");
        if m.request_to_result.is_empty() {
            let _ = writeln!(out, "  (no data)");
        } else {
            let mut latencies: Vec<f64> = m.request_to_result.iter().map(|d| dur_ms(*d)).collect();
            latencies.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let count = latencies.len();
            let sum: f64 = latencies.iter().sum();
            let avg = sum / count as f64;
            let min = latencies[0];
            let max = latencies[count - 1];
            let p95_idx = ((count as f64 * 0.95) as usize).min(count - 1);
            let p95 = latencies[p95_idx];
            let _ = writeln!(out, "  Count:  {}", count);
            let _ = writeln!(out, "  Min:    {:.2} ms", min);
            let _ = writeln!(out, "  Avg:    {:.2} ms", avg);
            let _ = writeln!(out, "  P95:    {:.2} ms", p95);
            let _ = writeln!(out, "  Max:    {:.2} ms", max);
        }
        let _ = writeln!(out);

        // Stall tracking
        if m.total_poll_frames > 0 {
            let _ = writeln!(out, "── Stall Frames ────────────────────────────────────────────");
            let _ = writeln!(out, "  Total polls:  {}", m.total_poll_frames);
            let _ = writeln!(out, "  Stall frames: {} ({:.1}%)",
                m.stall_frames,
                (m.stall_frames as f64 / m.total_poll_frames as f64) * 100.0);
            let _ = writeln!(out);
        }

        let _ = writeln!(out, "================================================================");
        out
    }

    /// Generate report and return as a C string for FFI.
    /// Caller must free with `voxel_profiler_free_report`.
    pub fn generate_report_cstr(&self) -> *mut c_char {
        let report = self.generate_report();
        match CString::new(report) {
            Ok(cstr) => cstr.into_raw(),
            Err(_) => {
                // Fallback: strip any interior NUL bytes
                let report = self.generate_report().replace('\0', "");
                CString::new(report)
                    .unwrap_or_else(|_| CString::new("(report generation failed)").unwrap())
                    .into_raw()
            }
        }
    }
}

/// Helper: Duration to milliseconds as f64.
fn dur_ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

/// Helper: min/avg/max of a u32 vec.
fn vec_stats(v: &[u32]) -> (u32, f64, u32) {
    if v.is_empty() {
        return (0, 0.0, 0);
    }
    let min = *v.iter().min().unwrap();
    let max = *v.iter().max().unwrap();
    let sum: u64 = v.iter().map(|&x| x as u64).sum();
    let avg = sum as f64 / v.len() as f64;
    (min, avg, max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profiler_disabled_by_default() {
        let p = StreamingProfiler::new(4);
        assert!(!p.is_enabled());
    }

    #[test]
    fn enable_disable_toggle() {
        let p = StreamingProfiler::new(4);
        p.set_enabled(true);
        assert!(p.is_enabled());
        p.set_enabled(false);
        assert!(!p.is_enabled());
    }

    #[test]
    fn session_lifecycle() {
        let p = StreamingProfiler::new(2);
        p.set_enabled(true);

        let id = p.begin_session("seed=42, chunk_size=16".to_string());
        assert_eq!(id, 1);

        // Record some chunk data
        let timings = ChunkTimings {
            region_density: Duration::from_millis(10),
            hermite: Duration::from_millis(5),
            dc_solve: Duration::from_millis(8),
            mesh_gen: Duration::from_millis(12),
            seam_pass: Duration::from_millis(3),
            coord_transform: Duration::from_millis(1),
            store_read_wait: Duration::from_millis(2),
            store_write_wait: Duration::from_millis(4),
            total: Duration::from_millis(45),
            was_slow_path: true,
            vertex_count: 1200,
            triangle_count: 800,
            section_count: 3,
            mesh_bytes: 48000,
            seam_quad_gen: Duration::ZERO,
            seam_mesh_retrieve: Duration::ZERO,
            seam_convert: Duration::ZERO,
            seam_candidates_tried: 0,
            seam_candidates_sent: 0,
            send_block: Duration::ZERO,
            coarse_skip: false,
        };

        p.record_chunk_with_coord(0, (0, 0, 0), timings, 5, 2);
        p.record_worker_idle(1, Duration::from_millis(100));
        p.end_session();

        let report = p.generate_report();
        assert!(report.contains("Session #1"));
        assert!(report.contains("region_density"));
        assert!(report.contains("Slow path:    1"));
        assert!(report.contains("seed=42"));
    }

    #[test]
    fn phase_stats_tracking() {
        let mut ps = PhaseStats::new();
        ps.record(Duration::from_millis(10));
        ps.record(Duration::from_millis(20));
        ps.record(Duration::from_millis(30));

        assert_eq!(ps.count, 3);
        assert_eq!(ps.min, Duration::from_millis(10));
        assert_eq!(ps.max, Duration::from_millis(30));
        assert_eq!(ps.avg(), Duration::from_millis(20));
        assert_eq!(ps.total, Duration::from_millis(60));
    }

    #[test]
    fn report_cstr_roundtrip() {
        let p = StreamingProfiler::new(1);
        p.set_enabled(true);
        let _ = p.begin_session("test".to_string());
        p.end_session();

        let ptr = p.generate_report_cstr();
        assert!(!ptr.is_null());
        // Free it properly
        unsafe { drop(CString::from_raw(ptr)); }
    }

    #[test]
    fn disabled_profiler_no_data() {
        let p = StreamingProfiler::new(2);
        // NOT enabled

        let timings = ChunkTimings {
            region_density: Duration::from_millis(10),
            hermite: Duration::from_millis(5),
            dc_solve: Duration::from_millis(8),
            mesh_gen: Duration::from_millis(12),
            seam_pass: Duration::from_millis(3),
            coord_transform: Duration::from_millis(1),
            store_read_wait: Duration::from_millis(2),
            store_write_wait: Duration::from_millis(4),
            total: Duration::from_millis(45),
            was_slow_path: false,
            vertex_count: 100,
            triangle_count: 50,
            section_count: 1,
            mesh_bytes: 3000,
            seam_quad_gen: Duration::ZERO,
            seam_mesh_retrieve: Duration::ZERO,
            seam_convert: Duration::ZERO,
            seam_candidates_tried: 0,
            seam_candidates_sent: 0,
            send_block: Duration::ZERO,
            coarse_skip: false,
        };

        p.record_chunk(0, timings, 0, 0);
        p.record_worker_idle(0, Duration::from_millis(50));
        p.record_stale_skip(1);

        // Should have no data since profiler was disabled
        let m = p.metrics.lock().unwrap();
        assert_eq!(m.total_chunk.count, 0);
    }

    #[test]
    fn request_to_result_latency() {
        let p = StreamingProfiler::new(1);
        p.set_enabled(true);
        let _ = p.begin_session("test".to_string());

        p.record_request((0, 0, 0));
        std::thread::sleep(Duration::from_millis(10));

        let timings = ChunkTimings {
            region_density: Duration::from_millis(5),
            hermite: Duration::from_millis(2),
            dc_solve: Duration::from_millis(3),
            mesh_gen: Duration::from_millis(4),
            seam_pass: Duration::from_millis(1),
            coord_transform: Duration::from_millis(1),
            store_read_wait: Duration::from_millis(1),
            store_write_wait: Duration::from_millis(1),
            total: Duration::from_millis(18),
            was_slow_path: false,
            vertex_count: 500,
            triangle_count: 300,
            section_count: 2,
            mesh_bytes: 15000,
            seam_quad_gen: Duration::ZERO,
            seam_mesh_retrieve: Duration::ZERO,
            seam_convert: Duration::ZERO,
            seam_candidates_tried: 0,
            seam_candidates_sent: 0,
            send_block: Duration::ZERO,
            coarse_skip: false,
        };

        p.record_chunk_with_coord(0, (0, 0, 0), timings, 0, 0);
        p.end_session();

        let m = p.metrics.lock().unwrap();
        assert_eq!(m.request_to_result.len(), 1);
        assert!(m.request_to_result[0] >= Duration::from_millis(10));
    }
}
