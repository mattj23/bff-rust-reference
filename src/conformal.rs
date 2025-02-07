use crate::mesh_structure::MeshStructure;
use crate::Result;
use faer::sparse::{SparseColMat, Triplet};
use std::collections::HashMap;
use std::f64::consts::PI;

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
) -> Result<SparseColMat<u32, f64>> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::*;
    use approx::assert_relative_eq;

    #[test]
    fn laplacian_abb() {
        let structure = get_test_structure();
        let boundary = structure.single_boundary_vertices().unwrap();

        let temp = cotan_laplacian_triplets(&structure).unwrap();
        let abb = slice_triplets_to_sparse(&boundary, &boundary, &temp).unwrap();
        let triplets = sparse_as_triplets(&abb);

        let expected = get_sparse_triplets("abb.coo");

        assert_triplets_eq!(triplets, expected);
    }

    #[test]
    fn laplacian_aib() {
        let structure = get_test_structure();
        let inner = structure.inner_vertices().unwrap();
        let boundary = structure.single_boundary_vertices().unwrap();

        let temp = cotan_laplacian_triplets(&structure).unwrap();
        let aib = slice_triplets_to_sparse(&inner, &boundary, &temp).unwrap();
        let triplets = sparse_as_triplets(&aib);

        let expected = get_sparse_triplets("aib.coo");

        assert_triplets_eq!(triplets, expected);
    }

    #[test]
    fn laplacian_aii() {
        let structure = get_test_structure();
        let inner = structure.inner_vertices().unwrap();

        let temp = cotan_laplacian_triplets(&structure).unwrap();
        let aii = slice_triplets_to_sparse(&inner, &inner, &temp).unwrap();
        let triplets = sparse_as_triplets(&aii);

        let expected = get_sparse_triplets("aii.coo");

        assert_triplets_eq!(triplets, expected);
    }

    #[test]
    fn cotan_laplacian_calc() {
        let structure = get_test_structure();
        let n = structure.vertices.len();

        let temp = cotan_laplacian_triplets(&structure).unwrap();
        let laplacian = SparseColMat::try_new_from_triplets(n, n, &temp).unwrap();
        let triplets = sparse_as_triplets(&laplacian);

        let expected = get_sparse_triplets("laplacian.coo");
        assert_triplets_eq!(triplets, expected);
    }

    #[test]
    fn boundary_edge_length_calc() {
        let structure = get_test_structure();
        let edge_lengths = boundary_edge_lengths(&structure).unwrap();

        let expected = get_float_vector("boundary_edge_lengths.floatvec");
        assert_eq!(edge_lengths.len(), expected.len());

        for (test, known) in edge_lengths.iter().zip(expected.iter()) {
            assert_relative_eq!(test, known, epsilon = 1e-6);
        }
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
