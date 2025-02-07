mod conformal;
mod layout;
pub mod mesh_structure;
pub mod serialize;

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

pub type Vector3 = nalgebra::SVector<f64, 3>;
pub type Point3 = nalgebra::Point3<f64>;

#[cfg(test)]
mod test_utils {
    use super::*;
    use crate::mesh_structure::MeshStructure;
    use crate::serialize::MeshData;
    use faer::sparse::SparseColMat;
    use std::io::Read;
    use faer::Mat;
    use zip::ZipArchive;

    const DATA_BYTES: &[u8] = include_bytes!("test_data.zip");

    macro_rules! assert_triplets_eq {
        ($a:ident, $b:ident) => {
            assert_eq!($a.len(), $b.len());
            for (i, (a, b)) in $a.iter().zip($b.iter()).enumerate() {
                assert_eq!(a.0, b.0, "Row mismatch at index {}", i);
                assert_eq!(a.1, b.1, "Column mismatch at index {}", i);
                assert_relative_eq!(a.2, b.2, max_relative = 1e-8);
            }
        };
    }

    use crate::conformal::{boundary_edge_lengths, calc_angle_defects, calc_face_angles, cotan_laplacian_triplets, dirichlet_boundary, laplacian_set};
    pub(crate) use assert_triplets_eq;
    use crate::layout::best_fit_curve;

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
        let mut triplets: Vec<(u32, u32, f64)> =
            rmp_serde::from_read(std::io::Cursor::new(bytes)).unwrap();
        triplets.sort_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));
        triplets
    }

    pub fn get_test_structure() -> MeshStructure {
        let (vertices, faces) = get_mesh_data();
        MeshStructure::new(vertices, faces).unwrap()
    }

    #[test]
    fn end_to_end() -> Result<()> {
        let mesh = get_test_structure();
        let n_vert = mesh.vertices.len();

        let i_inner = mesh.inner_vertices()?;
        let i_bound = mesh.single_boundary_vertices()?;

        let face_angles = calc_face_angles(&mesh)?;
        let angle_defects = calc_angle_defects(n_vert, &i_bound, &face_angles, &mesh.faces)?;

        let triplets =
            cotan_laplacian_triplets(&face_angles, n_vert, &mesh.edges, &mesh.face_edges)?;
        let (a, aii, aib, abb) = laplacian_set(n_vert, &i_inner, &i_bound, &triplets)?;

        let aii_lu = aii.sp_lu()?;
        let ub = Mat::<f64>::zeros(i_bound.len(), 1);
        let im_k = dirichlet_boundary(
            &ub,
            &aii_lu,
            &aib,
            &abb,
            &mesh.inner_vertices()?,
            &mesh.single_boundary_vertices()?,
            &angle_defects,
        )?;

        let boundary_edge_len = boundary_edge_lengths(&mesh)?;

        // let uvb = best_fit_curve()

        Ok(())
    }
}
