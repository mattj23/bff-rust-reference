pub mod serialize;
pub mod mesh_structure;

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

pub type Vector3 = nalgebra::SVector<f64, 3>;
pub type Point3 = nalgebra::Point3<f64>;


#[cfg(test)]
mod test_utils {
    use std::io::Read;
    use zip::ZipArchive;
    use crate::Point3;
    use crate::serialize::MeshData;

    const DATA_BYTES: &[u8] = include_bytes!("test_data.zip");

    fn get_file_bytes(file_name: &str) -> Vec<u8> {
        let mut archive = ZipArchive::new(std::io::Cursor::new(DATA_BYTES)).unwrap();
        let mut file = archive.by_name(file_name).unwrap();
        let mut contents = Vec::new();
        file.read_to_end(&mut contents).unwrap();
        contents
    }

    pub fn get_mesh_data() -> (Vec<Point3>, Vec<[u32; 3]>) {
        let bytes = get_file_bytes("hyperboloid.msgpack");
        let mesh_data: MeshData = rmp_serde::from_read(std::io::Cursor::new(bytes)).unwrap();

        (mesh_data.vertices, mesh_data.faces)
    }

    pub fn get_int_matrix(file_name: &str) -> Vec<Vec<i64>> {
        let bytes = get_file_bytes(file_name);
        rmp_serde::from_read(std::io::Cursor::new(bytes)).unwrap()
    }

    #[test]
    fn test_data_zip() {
        let (vertices, faces) = get_mesh_data();
        assert_eq!(vertices.len(), 320);
        assert_eq!(faces.len(), 571);
    }

}