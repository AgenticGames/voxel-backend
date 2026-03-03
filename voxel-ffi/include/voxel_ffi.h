#ifndef VOXEL_FFI_H
#define VOXEL_FFI_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Basic vector types ── */

typedef struct {
    float x, y, z;
} FfiVec3;

typedef struct {
    int32_t x, y, z;
} FfiChunkCoord;

/* ── Mesh data (SoA layout, pointers owned by Rust) ── */

typedef struct {
    FfiVec3*  positions;
    FfiVec3*  normals;
    uint8_t*  material_ids;
    uint32_t  vertex_count;
    uint32_t* indices;
    uint32_t  index_count;
} FfiMeshData;

/* ── Mined material counts ── */

typedef struct {
    uint32_t counts[20];
} FfiMinedMaterials;

/* ── Result type enum (repr(u8) in Rust) ── */

#define FFI_RESULT_NONE        0
#define FFI_RESULT_CHUNK_MESH  1
#define FFI_RESULT_MINE_RESULT 2
#define FFI_RESULT_ERROR       3

/* ── Poll result (heap-allocated, must free with voxel_free_result) ── */

typedef struct {
    uint8_t          result_type;  /* FfiResultType (repr(u8)) */
    FfiChunkCoord    chunk;
    FfiMeshData      mesh;
    FfiMinedMaterials mined;
    uint64_t         generation;
} FfiResult;

/* ── Engine configuration ── */

typedef struct {
    uint64_t seed;
    uint32_t chunk_size;
    uint32_t worker_threads;
    float    world_scale;
    float    max_edge_length;
    /* Noise */
    double   cavern_frequency;
    double   cavern_threshold;
    uint32_t detail_octaves;
    double   detail_persistence;
    double   warp_amplitude;
    /* Worm */
    float    worms_per_region;
    float    worm_radius_min;
    float    worm_radius_max;
    float    worm_step_length;
    uint32_t worm_max_steps;
    float    worm_falloff_power;
} FfiEngineConfig;

/* ── Mine request ── */

typedef struct {
    float   world_x, world_y, world_z;
    float   radius;
    uint8_t mode;    /* 0=sphere, 1=peel */
    float   normal_x, normal_y, normal_z;
} FfiMineRequest;

/* ── Engine statistics ── */

typedef struct {
    uint32_t chunks_loaded;
    uint32_t pending_requests;
    uint32_t completed_results;
    uint32_t worker_threads_active;
} FfiEngineStats;

/* ── Lifecycle ── */

void*           voxel_create_engine(const FfiEngineConfig* config);
void            voxel_destroy_engine(void* engine);

/* ── Request submission (non-blocking, returns 1=success, 0=queue full) ── */

uint32_t        voxel_request_generate(void* engine, FfiChunkCoord chunk);
uint32_t        voxel_request_generate_batch(void* engine, const FfiChunkCoord* chunks, uint32_t count);
uint32_t        voxel_request_mine(void* engine, const FfiMineRequest* request);
uint32_t        voxel_request_unload(void* engine, FfiChunkCoord chunk);
void            voxel_cancel_chunk(void* engine, FfiChunkCoord chunk);

/* ── Polling (called each tick, non-blocking) ── */

FfiResult*      voxel_poll_result(void* engine);
void            voxel_free_result(FfiResult* result);

/* ── Query ── */

FfiEngineStats  voxel_get_stats(void* engine);
void            voxel_update_config(void* engine, const FfiEngineConfig* config);

#ifdef __cplusplus
}
#endif

#endif /* VOXEL_FFI_H */
