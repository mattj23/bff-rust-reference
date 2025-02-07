use crate::mesh_structure::MeshStructure;
use crate::Result;
use faer::sparse::{SparseColMat, Triplet};
use std::collections::HashMap;
use std::f64::consts::PI;

type SparseMat = SparseColMat<u32, f64>;

fn boundary_edge_lengths(mesh: &MeshStructure) -> Result<Vec<f64>> {
    Ok(mesh
        .single_boundary_vertices()?
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let next = mesh.boundaries[0][(i + 1) % mesh.boundaries[0].len()];
            (mesh.vertices[v as usize] - mesh.vertices[next as usize]).norm()
        })
        .collect())
}

/// Calculate the angles of the faces in the mesh. Each element `i` of the returned list corresponds
/// with the face at `mesh.faces[i]`, while the value at `j` for that face corresponds with the
/// angle formed at the vertex opposite to the `j`th edge `mesh.face_edges[i][j]`.
fn face_angles(mesh: &MeshStructure) -> Result<Vec<[f64; 3]>> {
    let mut angles: Vec<[f64; 3]> = Vec::with_capacity(mesh.faces.len());

    for face_indices in mesh.face_edges.iter() {
        let a = mesh.edge_lengths[face_indices[0] as usize];
        let b = mesh.edge_lengths[face_indices[1] as usize];
        let c = mesh.edge_lengths[face_indices[2] as usize];

        // Check for degenerate faces
        let face_angles = if a > b + c {
            [PI, 0.0, 0.0]
        } else if b > a + c {
            [0.0, PI, 0.0]
        } else if c > a + b {
            [0.0, 0.0, PI]
        } else {
            let cos_a = (b.powi(2) + c.powi(2) - a.powi(2)) / (2.0 * b * c);
            let cos_b = (a.powi(2) + c.powi(2) - b.powi(2)) / (2.0 * a * c);
            let cos_c = (a.powi(2) + b.powi(2) - c.powi(2)) / (2.0 * a * b);
            [cos_a.acos(), cos_b.acos(), cos_c.acos()]
        };

        angles.push(face_angles);
    }

    Ok(angles)
}

fn angle_defects(mesh: &MeshStructure) -> Result<Vec<f64>> {
    let mut thetas = vec![2.0 * PI; mesh.vertices.len()];
    for &i in mesh.single_boundary_vertices()? {
        thetas[i as usize] = PI;
    }

    let angles = face_angles(mesh)?;
    for (face, angles) in mesh.faces.iter().zip(angles.iter()) {
        thetas[face[0] as usize] -= angles[0];
        thetas[face[1] as usize] -= angles[1];
        thetas[face[2] as usize] -= angles[2];
    }

    Ok(thetas)
}

fn cotan_laplacian_triplets(mesh: &MeshStructure) -> Result<Vec<Triplet<u32, u32, f64>>> {
    let cotans = face_angles(mesh)?
        .iter()
        .map(|angles| {
            [
                1.0 / angles[0].tan(),
                1.0 / angles[1].tan(),
                1.0 / angles[2].tan(),
            ]
        })
        .collect::<Vec<[f64; 3]>>();

    let mut values = vec![0.0; mesh.edges.len()];
    for (face, cotan) in mesh.face_edges.iter().zip(cotans.iter()) {
        for (i, &edge) in face.iter().enumerate() {
            values[edge as usize] += cotan[i];
        }
    }

    // Multiply by 0.5 to account for the fact that each edge is shared by two faces
    for value in values.iter_mut() {
        *value *= 0.5;
    }

    // Prepare the diagonal values
    let mut diagonals = vec![0.0; mesh.vertices.len()];
    for (edge, &value) in mesh.edges.iter().zip(values.iter()) {
        diagonals[edge[0] as usize] += value;
        diagonals[edge[1] as usize] += value;
    }

    // Build the sparse matrix
    let mut triplets = Vec::new();
    for (i, &value) in diagonals.iter().enumerate() {
        triplets.push(Triplet::new(i as u32, i as u32, value));
    }

    for (edge, &value) in mesh.edges.iter().zip(values.iter()) {
        triplets.push(Triplet::new(edge[0], edge[1], -value));
        triplets.push(Triplet::new(edge[1], edge[0], -value));
    }

    Ok(triplets)
}

fn slice_triplets_to_sparse(
    rows: &[u32],
    cols: &[u32],
    triplets: &[Triplet<u32, u32, f64>],
) -> Result<SparseMat> {
    let row_check: HashMap<u32, u32> = rows
        .iter()
        .enumerate()
        .map(|(i, &v)| (v, i as u32))
        .collect();
    let col_check: HashMap<u32, u32> = cols
        .iter()
        .enumerate()
        .map(|(i, &v)| (v, i as u32))
        .collect();

    let updated = triplets
        .iter()
        .filter_map(|t| {
            if let (Some(&row), Some(&col)) = (row_check.get(&t.row), col_check.get(&t.col)) {
                Some(Triplet::new(row, col, t.val))
            } else {
                None
            }
        })
        .collect::<Vec<_>>();

    SparseColMat::try_new_from_triplets(rows.len(), cols.len(), &updated).map_err(|e| e.into())
}

/// Calculate the set of sparse laplacian matrices A, AII, AIB, and ABB for the given data.
///
/// # Arguments
///
/// * `n`: the number of vertices in the mesh
/// * `i_inner`: the indices of the inner vertices, in their original order
/// * `i_bound`: the indices of the boundary vertices, in their original order
/// * `triplets`: the total set of triplets for the cotangent laplacian matrix
///
/// returns: Result<(SparseColMat<u32, f64, usize, usize>, SparseColMat<u32, f64, usize, usize>, SparseColMat<u32, f64, usize, usize>, SparseColMat<u32, f64, usize, usize>), Box<dyn Error, Global>>
pub fn laplacian_set(n: usize, i_inner: &[u32], i_bound: &[u32], triplets: &[Triplet<u32, u32, f64>]) -> Result<(SparseMat, SparseMat, SparseMat, SparseMat)> {
    // let triplets = cotan_laplacian_triplets(&mesh)?;

    // The complete cotangent laplacian matrix A
    let a = SparseColMat::try_new_from_triplets(n, n, &triplets)?;
    let aii = slice_triplets_to_sparse(&i_inner, &i_inner, &triplets)?;
    let aib = slice_triplets_to_sparse(&i_inner, &i_bound, &triplets)?;
    let abb = slice_triplets_to_sparse(&i_bound, &i_bound, &triplets)?;

    Ok((a, aii, aib, abb))
}

pub fn dirichlet_boundary(mesh: &MeshStructure) -> Result<Vec<f64>> {
    todo!()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::*;
    use approx::assert_relative_eq;
    use faer::linalg::solvers::Solve;
    use faer::Mat;
    use std::ops::Mul;

    fn laplacian_mock() -> Result<(SparseMat, SparseMat, SparseMat, SparseMat)> {
        let mesh = get_test_structure();
        let inner_indices = mesh.inner_vertices()?;
        let boundary_indices = mesh.single_boundary_vertices()?;
        let triplets = cotan_laplacian_triplets(&mesh)?;

        laplacian_set(mesh.vertices.len(), &inner_indices, &boundary_indices, &triplets)
    }

    #[test]
    fn dirichlet_boundary_im_k() -> Result<()> {
        let mesh = get_test_structure();
        let (_, aii, aib, abb) = laplacian_mock()?;

        let all_defects = angle_defects(&mesh)?;
        let inner_defects = mesh
            .inner_vertices()?
            .iter()
            .map(|i| all_defects[*i as usize])
            .collect::<Vec<_>>();

        let defects = Mat::from_fn(inner_defects.len(), 1, |i, _| inner_defects[i]);

        let ub = Mat::<f64>::zeros(mesh.single_boundary_vertices()?.len(), 1);
        let value = &defects + &aib * &ub;

        let aii_lu = aii.sp_lu().unwrap();
        let ui = -aii_lu.solve(&value);

        let h = -&aib.transpose() * &ui - &abb * &ub;
        let h = h.row_iter().map(|r| r[0]).collect::<Vec<f64>>();

        let im_k = mesh.single_boundary_vertices()?.iter().zip(h.iter())
            .map(|(&vi, &hv)| all_defects[vi as usize] - hv)
            .collect::<Vec<f64>>();

        let expected = get_float_vector("dirichlet_im_k.floatvec");

        assert_eq!(im_k.len(), expected.len());
        for (test, known) in im_k.iter().zip(expected.iter()) {
            assert_relative_eq!(test, known, epsilon = 1e-6);
        }

        Ok(())
    }

    #[test]
    fn dirichlet_h() -> Result<()>{
        let mesh = get_test_structure();
        let (a, aii, aib, abb) = laplacian_mock()?;

        let all_defects = angle_defects(&mesh).unwrap();
        let inner_defects = mesh
            .inner_vertices()
            .unwrap()
            .iter()
            .map(|i| all_defects[*i as usize])
            .collect::<Vec<_>>();

        let defects = Mat::from_fn(inner_defects.len(), 1, |i, _| inner_defects[i]);

        let ub = Mat::<f64>::zeros(mesh.single_boundary_vertices().unwrap().len(), 1);
        let value = &defects + &aib * &ub;

        let aii_lu = aii.sp_lu().unwrap();
        let ui = -aii_lu.solve(&value);

        let h = -&aib.transpose() * &ui - &abb * &ub;
        let check: Vec<f64> = h.row_iter().map(|r| r[0]).collect();

        let expected = get_float_vector("dirichlet_h.floatvec");

        assert_eq!(check.len(), expected.len());
        for (test, known) in check.iter().zip(expected.iter()) {
            assert_relative_eq!(test, known, epsilon = 1e-6);
        }

        Ok(())
    }


    #[test]
    fn dirichlet_ui() -> Result<()>{
        let mesh = get_test_structure();
        let (a, aii, aib, abb) = laplacian_mock()?;

        let all_defects = angle_defects(&mesh).unwrap();
        let inner_defects = mesh
            .inner_vertices()
            .unwrap()
            .iter()
            .map(|i| all_defects[*i as usize])
            .collect::<Vec<_>>();

        let defects = Mat::from_fn(inner_defects.len(), 1, |i, _| inner_defects[i]);

        let ub = Mat::<f64>::zeros(mesh.single_boundary_vertices().unwrap().len(), 1);
        let value = &defects + &aib.mul(&ub);

        let aii_lu = aii.sp_lu().unwrap();
        let ui = -aii_lu.solve(&value);
        let check: Vec<f64> = ui.row_iter().map(|r| r[0]).collect();

        let expected = get_float_vector("dirichlet_ui.floatvec");

        assert_eq!(check.len(), expected.len());
        for (test, known) in check.iter().zip(expected.iter()) {
            assert_relative_eq!(test, known, epsilon = 1e-6);
        }

        Ok(())
    }

    #[test]
    fn laplacian_abb() -> Result<()>{
        let (_, _, _, abb) = laplacian_mock()?;
        let triplets = sparse_as_triplets(&abb);
        let expected = get_sparse_triplets("abb.coo");

        assert_triplets_eq!(triplets, expected);

        Ok(())
    }

    #[test]
    fn laplacian_aib() -> Result<()>{
        let (_, _, aib, _) = laplacian_mock()?;
        let triplets = sparse_as_triplets(&aib);
        let expected = get_sparse_triplets("aib.coo");

        assert_triplets_eq!(triplets, expected);

        Ok(())
    }

    #[test]
    fn laplacian_aii() -> Result<()>{
        let (_, aii, _, _) = laplacian_mock()?;
        let triplets = sparse_as_triplets(&aii);
        let expected = get_sparse_triplets("aii.coo");

        assert_triplets_eq!(triplets, expected);

        Ok(())
    }

    #[test]
    fn cotan_laplacian_calc() -> Result<()> {
        let (a, _, _, _) = laplacian_mock()?;
        let triplets = sparse_as_triplets(&a);
        let expected = get_sparse_triplets("laplacian.coo");

        assert_triplets_eq!(triplets, expected);

        Ok(())
    }

    #[test]
    fn boundary_edge_length_calc() -> Result<()>{
        let structure = get_test_structure();
        let edge_lengths = boundary_edge_lengths(&structure)?;

        let expected = get_float_vector("boundary_edge_lengths.floatvec");
        assert_eq!(edge_lengths.len(), expected.len());

        for (test, known) in edge_lengths.iter().zip(expected.iter()) {
            assert_relative_eq!(test, known, epsilon = 1e-6);
        }

        Ok(())
    }

    #[test]
    fn triangle_face_angles() {
        let structure = get_test_structure();
        let angles = face_angles(&structure).unwrap();

        let expected_data = get_float_matrix("tri_angles.floatmat");
        let expected: Vec<[f64; 3]> = expected_data
            .iter()
            .map(|row| [row[0], row[1], row[2]])
            .collect();

        assert_eq!(angles.len(), expected.len());
        for (test, known) in angles.iter().zip(expected.iter()) {
            for (t, k) in test.iter().zip(known.iter()) {
                assert_relative_eq!(t, k, epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn angle_defects_calc() {
        let structure = get_test_structure();
        let expected = get_float_vector("angle_defects.floatvec");

        let defects = angle_defects(&structure).unwrap();

        assert_eq!(defects.len(), expected.len());

        for (test, known) in defects.iter().zip(expected.iter()) {
            assert_relative_eq!(test, known, epsilon = 1e-6);
        }
    }
}
