use std::path::Path;
use crate::Result;
use rmp_serde::from_read;
use serde::{Deserialize, Serialize};
use crate::Point3;

#[derive(Debug, Clone, Deserialize)]
pub struct MeshData {
    pub vertices: Vec<Point3>,
    pub faces: Vec<[u32; 3]>,
}

impl MeshData {
    pub fn load(path: &Path) -> Result<Self> {
        let file = std::fs::File::open(path)?;
        let reader = std::io::BufReader::new(file);
        Ok(from_read(reader)?)
    }
}