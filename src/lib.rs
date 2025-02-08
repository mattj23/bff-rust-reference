use faer::sparse::SparseColMat;
use faer::Mat;

mod conformal;
mod layout;
pub mod mesh_structure;
pub mod serialize;

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

pub type Vector3 = nalgebra::SVector<f64, 3>;
pub type Point3 = nalgebra::Point3<f64>;
pub type Point2 = nalgebra::Point2<f64>;
pub type SparseMat = SparseColMat<u32, f64>;

// Quick helper for making a single row matrix from slice
pub fn single_row_matrix(data: &[f64]) -> Mat<f64> {
    Mat::from_fn(1, data.len(), |_, j| data[j])
}

// Quick helper for making a single column matrix from slice
pub fn single_col_matrix(data: &[f64]) -> Mat<f64> {
    Mat::from_fn(data.len(), 1, |i, _| data[i])
}

pub fn invert_2x2(m: &Mat<f64>) -> Result<Mat<f64>> {
    if m.nrows() != 2 || m.ncols() != 2 {
        return Err("Matrix is not 2x2".into());
    }

    let det = m[(0, 0)] * m[(1, 1)] - m[(0, 1)] * m[(1, 0)];
    if det == 0.0 {
        return Err("Matrix is singular".into());
    }

    let inv_det = 1.0 / det;
    let mut result = Mat::zeros(2, 2);
    result[(0, 0)] = m[(1, 1)] * inv_det;
    result[(0, 1)] = -m[(0, 1)] * inv_det;
    result[(1, 0)] = -m[(1, 0)] * inv_det;
    result[(1, 1)] = m[(0, 0)] * inv_det;

    Ok(result)
}

#[cfg(test)]
mod test_utils {
    use super::*;
    use crate::mesh_structure::MeshStructure;
    use crate::serialize::MeshData;
    use approx::assert_relative_eq;
    use faer::sparse::SparseColMat;
    use faer::Mat;
    use std::io::Read;
    use zip::ZipArchive;

    const DATA_BYTES: &[u8] = include_bytes!("test_data.zip");

    macro_rules! assert_triplets_eq {
        ($a:ident, $b:ident) => {
            assert_eq!($a.len(), $b.len());
            for (i, (a, b)) in $a.iter().zip($b.iter()).enumerate() {
                assert_eq!(a.0, b.0, "Row mismatch at index {}", i);
                assert_eq!(a.1, b.1, "Column mismatch at index {}", i);
                assert_relative_eq!(a.2, b.2, epsilon = 1e-6);
            }
        };
    }
    pub(crate) use assert_triplets_eq;

    macro_rules! assert_matrices_eq {
        ($a:ident, $b:ident, $rel:expr) => {
            assert_eq!($a.len(), $b.len());
            for (ar, br) in $a.iter().zip($b.iter()) {
                assert_eq!(ar.len(), br.len());
                for (ac, bc) in ar.iter().zip(br.iter()) {
                    assert_relative_eq!(ac, bc, epsilon = $rel);
                }
            }
        };
    }
    pub(crate) use assert_matrices_eq;

    macro_rules! assert_vectors_eq {
        ($a:ident, $b:ident) => {
            assert_eq!($a.len(), $b.len());
            for (a, b) in $a.iter().zip($b.iter()) {
                assert_relative_eq!(a, b, epsilon = 1e-6);
            }
        };
    }
    pub(crate) use assert_vectors_eq;

    use crate::conformal::{
        boundary_edge_lengths, calc_angle_defects, calc_face_angles, cotan_laplacian_triplets,
        dirichlet_boundary, laplacian_set,
    };
    use crate::layout::{best_fit_curve, extend_curve};

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
        let a_lu = a.sp_lu()?;
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

        let _uvb = best_fit_curve(&ub, &im_k, &boundary_edge_len)?;

        let uv = extend_curve(&a_lu, &aii_lu, &aib, &mesh.vertices, &i_bound, &i_inner)?;

        let expected_data = get_float_matrix("layout_uv.floatmat");
        let expected = expected_data
            .iter()
            .map(|r| Point2::new(r[0], r[1]))
            .collect::<Vec<_>>();

        assert_eq!(uv.len(), expected.len());
        for (a, b) in uv.iter().zip(expected.iter()) {
            assert_relative_eq!(a, b, max_relative = 1e-8);
        }

        Ok(())
    }
}
