use crate::{dist, Result};
use std::collections::{HashMap, HashSet};

pub struct MeshStructure {
    /// A list of vertices in the mesh. The order of the vertices is unimportant on its own, but
    /// the elements of `faces` and `edges` refer to the indices in this list.
    pub vertices: Vec<[f64; 3]>,

    /// A list of faces in the mesh. Each face consists of three indices into the `vertices` list,
    /// which are the three vertices of the face in counter-clockwise order.
    pub faces: Vec<[u32; 3]>,

    /// A list of edges in the mesh. Each edge consists of two indices into the `vertices` list,
    /// which are the two vertices of the edge.
    pub edges: Vec<[u32; 2]>,

    /// A list of the lengths of each edge in the mesh. The order of the lengths is the same as the
    /// order of the edges in the `edges` list.
    pub edge_lengths: Vec<f64>,

    /// A list of edges associated with each face. A face at index `i` in `face_edges` corresponds
    /// with the face at index `i` in `faces`. The `faces` list references the three vertices,
    /// while the `face_edges` list references the three edges of the face.
    pub face_edges: Vec<[u32; 3]>,

    /// A list of the different boundaries in the mesh. Each boundary is a list of indices into the
    /// vertices list that form a loop.
    pub boundaries: Vec<Vec<u32>>,
}

impl MeshStructure {
    pub fn new(vertices: Vec<[f64; 3]>, faces: Vec<[u32; 3]>) -> Result<Self> {
        let (edges, face_edges, boundaries) = identify_edges(&faces)?;

        let edge_lengths = edges
            .iter()
            .map(|[i, j]| dist(&vertices[*i as usize], &vertices[*j as usize]))
            .collect();

        Ok(Self {
            vertices,
            faces,
            edges,
            edge_lengths,
            face_edges,
            boundaries,
        })
    }

    pub fn single_boundary_vertices(&self) -> Result<&[u32]> {
        if self.boundaries.len() != 1 {
            return Err("Mesh must have exactly one boundary".into());
        }

        Ok(&self.boundaries[0])
    }

    pub fn inner_vertices(&self) -> Result<Vec<u32>> {
        let boundary_set: HashSet<u32> = self.single_boundary_vertices()?.iter().copied().collect();
        let inner: Vec<u32> = (0..self.vertices.len() as u32)
            .filter(|&i| !boundary_set.contains(&i))
            .collect();
        Ok(inner)
    }
}

/// Given an edge, return a key that can be used to identify the vertices that are connected by the
/// edge without regard to the order of the vertices.
fn edge_key(edge: &[u32; 2]) -> [u32; 2] {
    let x = edge[0].min(edge[1]);
    let y = edge[0].max(edge[1]);
    [x, y]
}

/// Generates a complete list of edges from the faces of a mesh, naively, and in the order that
/// the faces are presented. The edges are not deduplicated nor their direction normalized. The
/// resulting list will be 3x the length of the original faces. Elements 0-2 will be from face 0,
/// elements 3-5 will be from face 1, and so on.
fn naive_edges(faces: &[[u32; 3]]) -> Vec<[u32; 2]> {
    let mut edges = Vec::new();
    for face in faces {
        edges.push([face[1], face[2]]);
        edges.push([face[2], face[0]]);
        edges.push([face[0], face[1]]);
    }
    edges
}

/// Given a reference list of all edges, return a sorted list of unique edges and the number of
/// times each edge appeared in the original list.
fn unique_edges(all_edges: &[[u32; 2]]) -> Vec<([u32; 2], usize)> {
    let mut unique = HashMap::new();
    for edge in all_edges {
        let key = edge_key(edge);
        let count = unique.entry(key).or_insert(0);
        *count += 1;
    }

    let mut unique_count: Vec<_> = unique.into_iter().collect();
    unique_count.sort();
    unique_count
}

fn boundary_loops(boundary_map: HashMap<u32, u32>) -> Vec<Vec<u32>> {
    let mut all_loops = Vec::new();
    let mut working = Vec::new();
    let mut queue: HashSet<u32> = boundary_map.keys().copied().collect();

    while !queue.is_empty() {
        if let Some(last_id) = working.last() {
            let next_id = boundary_map[last_id];
            queue.remove(&next_id);

            if *working.first().unwrap() == next_id {
                working.reverse();

                // TODO: Remove this operation in a real implementation
                // Roll the loop to start at the smallest index and move it into the list of loops.
                // The roll operation isn't necessary for the algorithm, but is necessary to have
                // output consistent with the reference implementation for testing.
                all_loops.push(roll_start_smallest(working));

                working = Vec::new();
            } else {
                working.push(next_id);
            }
        } else {
            let start_id = *queue.iter().next().unwrap();
            working.push(start_id);
        }
    }

    all_loops
}

fn roll_start_smallest(mut id_loop: Vec<u32>) -> Vec<u32> {
    let min_index = id_loop
        .iter()
        .enumerate()
        .min_by_key(|(_, &id)| id)
        .map(|(i, _)| i)
        .unwrap();
    id_loop.rotate_left(min_index);
    id_loop
}

fn identify_edges(faces: &[[u32; 3]]) -> Result<(Vec<[u32; 2]>, Vec<[u32; 3]>, Vec<Vec<u32>>)> {
    // The direct edges are the edges that are directly defined by the faces, kept in the same
    // order as they are defined in the faces.
    let direct_edges = naive_edges(faces);

    // We need to identify the unique edges, and put them into a sorted order where we can still
    // map from an original face's edge to the unique edge
    let unique_edge_count = unique_edges(&direct_edges);

    // Boundary edges are edges that only appear once in the mesh. Non-manifold edges are ones
    // that appear more than twice. Any non-manifold edges will cause this function to return an
    // error
    if unique_edge_count.iter().any(|(_, count)| *count > 2) {
        return Err("Non-manifold edges detected".into());
    }

    // Now we can create a mapping from the original edge to the corresponding unique edge
    let to_unique_index: HashMap<[u32; 2], usize> = unique_edge_count
        .iter()
        .enumerate()
        .map(|(i, (edge, _))| (*edge, i))
        .collect();

    // Let's remap the face edges to the unique edges and build the boundary map at the same time
    let mut boundary_map = HashMap::new();
    let mut face_edges = Vec::new();
    for face_chunk in direct_edges.chunks(3) {
        let i0 = to_unique_index[&edge_key(&face_chunk[0])];
        let i1 = to_unique_index[&edge_key(&face_chunk[1])];
        let i2 = to_unique_index[&edge_key(&face_chunk[2])];
        face_edges.push([i0 as u32, i1 as u32, i2 as u32]);

        if unique_edge_count[i0].1 == 1 {
            boundary_map.insert(face_chunk[0][0], face_chunk[0][1]);
        }
        if unique_edge_count[i1].1 == 1 {
            boundary_map.insert(face_chunk[1][0], face_chunk[1][1]);
        }
        if unique_edge_count[i2].1 == 1 {
            boundary_map.insert(face_chunk[2][0], face_chunk[2][1]);
        }
    }

    let loops = boundary_loops(boundary_map);
    let edges = unique_edge_count.iter().map(|(edge, _)| *edge).collect();

    Ok((edges, face_edges, loops))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::*;

    #[test]
    fn boundary_loops() {
        let (_vertices, faces) = get_mesh_data();
        let (_, _, loops) = identify_edges(&faces).unwrap();

        let expected_data = get_int_matrix("boundary_loops.intmat");
        let expected = expected_data
            .iter()
            .map(|row| row.iter().map(|&x| x as u32).collect::<Vec<_>>())
            .collect::<Vec<_>>();

        assert_eq!(loops, expected);
    }

    #[test]
    fn remapped_faces() {
        let (_vertices, faces) = get_mesh_data();
        let (_, face_edges, _) = identify_edges(&faces).unwrap();

        let expected_data = get_int_matrix("face_edges.intmat");
        let expected = expected_data
            .iter()
            .map(|row| [row[0] as u32, row[1] as u32, row[2] as u32])
            .collect::<Vec<_>>();

        assert_eq!(face_edges, expected);
    }

    #[test]
    fn unique_edge_identification() {
        let (_vertices, faces) = get_mesh_data();
        let expected_data = get_int_matrix("edges.intmat");
        let edges = naive_edges(&faces);
        let unique = unique_edges(&edges);

        // Reshape these into a more convenient format for comparison
        let expected = expected_data
            .iter()
            .map(|row| [row[0] as u32, row[1] as u32])
            .collect::<Vec<_>>();

        let test = unique.iter().map(|(edge, _)| *edge).collect::<Vec<_>>();
        assert_eq!(test, expected);
    }

    #[test]
    fn unique_edge_count() {
        let (_vertices, faces) = get_mesh_data();
        let expected = vec![
            2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            2, 2, 2, 2, 2, 2, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            2, 2, 2, 2, 2, 2, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 1, 2, 2, 2, 2, 2,
            2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2,
            2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2,
            2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2,
            2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2,
            2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 1, 1, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 2, 2, 2, 2, 1, 1, 1,
        ];
        let edges = naive_edges(&faces);
        let unique = unique_edges(&edges);
        let counts = unique.iter().map(|(_, count)| *count).collect::<Vec<_>>();

        assert_eq!(counts, expected);
    }
}
