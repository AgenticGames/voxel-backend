use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestReport {
    pub total_seeds: u32,
    pub passed: u32,
    pub failed: u32,
    /// All seed results (both passed and failed)
    pub results: Vec<FailureDetail>,
    /// Legacy: kept for backwards compat with older reports
    #[serde(default)]
    pub failures: Vec<FailureDetail>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailureDetail {
    pub seed: u64,
    #[serde(default)]
    pub passed: bool,
    pub reason: String,
    pub obj_path: Option<String>,
}

impl TestReport {
    pub fn pass_rate(&self) -> f64 {
        if self.total_seeds == 0 {
            return 0.0;
        }
        self.passed as f64 / self.total_seeds as f64
    }

    /// Serialize the report to JSON string
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|e| {
            format!("{{\"error\": \"Failed to serialize report: {}\"}}", e)
        })
    }

    /// Write the report to a JSON file
    pub fn write_to_file(&self, path: &str) -> std::io::Result<()> {
        let json = self.to_json();
        std::fs::write(path, json)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pass_rate_all_pass() {
        let report = TestReport {
            total_seeds: 10,
            passed: 10,
            failed: 0,
            results: Vec::new(),
            failures: Vec::new(),
        };
        assert!((report.pass_rate() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_pass_rate_some_fail() {
        let report = TestReport {
            total_seeds: 10,
            passed: 7,
            failed: 3,
            results: Vec::new(),
            failures: Vec::new(),
        };
        assert!((report.pass_rate() - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_pass_rate_none() {
        let report = TestReport {
            total_seeds: 0,
            passed: 0,
            failed: 0,
            results: Vec::new(),
            failures: Vec::new(),
        };
        assert!((report.pass_rate() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_to_json() {
        let report = TestReport {
            total_seeds: 2,
            passed: 1,
            failed: 1,
            results: vec![
                FailureDetail {
                    seed: 0,
                    passed: true,
                    reason: String::new(),
                    obj_path: Some("seed_0.obj".to_string()),
                },
                FailureDetail {
                    seed: 42,
                    passed: false,
                    reason: "mesh validation failed".to_string(),
                    obj_path: Some("seed_42.obj".to_string()),
                },
            ],
            failures: Vec::new(),
        };
        let json = report.to_json();
        assert!(json.contains("\"total_seeds\": 2"));
        assert!(json.contains("\"seed\": 42"));
        assert!(json.contains("mesh validation failed"));
    }
}
