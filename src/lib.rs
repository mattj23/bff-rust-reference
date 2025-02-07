pub mod serialize;
pub mod mesh_structure;
mod conformal;

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

pub type Vector3 = nalgebra::SVector<f64, 3>;
pub type Point3 = nalgebra::Point3<f64>;


#[cfg(test)]
mod test_utils {
    use std::io::Read;
    use faer::sparse::SparseColMat;
    use zip::ZipArchive;
    use crate::mesh_structure::MeshStructure;
    use crate::Point3;
    use crate::serialize::MeshData;

    const DATA_BYTES: &[u8] = include_bytes!("test_data.zip");

    pub fn sparse_as_triplets(sparse: &SparseColMat<u32, f64>) -> Vec<(u32, u32, f64)> {
        let mut triplets = Vec::new();
        let idx = sparse.row_idx().to_vec();
        let val = sparse.val().to_vec();

        // Expand the compressed column format into a full length vector
        let mut ptr = Vec::new();
        for i in 0..sparse.col_ptr().len() - 1 {
            let start = sparse.col_ptr()[i] as usize;
            let end = sparse.col_ptr()[i + 1] as usize;
            ptr.extend(std::iter::repeat(i as u32).take(end - start));
        }

        for (i, &value) in val.iter().enumerate() {
            triplets.push((idx[i], ptr[i], value));
        }
        triplets.sort_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));
        triplets
    }

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

    pub fn get_float_matrix(file_name: &str) -> Vec<Vec<f64>> {
        let bytes = get_file_bytes(file_name);
        rmp_serde::from_read(std::io::Cursor::new(bytes)).unwrap()
    }

    pub fn get_float_vector(file_name: &str) -> Vec<f64> {
        let bytes = get_file_bytes(file_name);
        rmp_serde::from_read(std::io::Cursor::new(bytes)).unwrap()
    }

    pub fn get_sparse_triplets(file_name: &str) -> Vec<(u32, u32, f64)> {
        let bytes = get_file_bytes(file_name);
        rmp_serde::from_read(std::io::Cursor::new(bytes)).unwrap()
    }

    pub fn get_test_structure() -> MeshStructure {
        let (vertices, faces) = get_mesh_data();
        MeshStructure::new(vertices, faces).unwrap()
    }

    #[test]
    fn test_data_zip() {
        let (vertices, faces) = get_mesh_data();
        assert_eq!(vertices.len(), 320);
        assert_eq!(faces.len(), 571);
    }

}