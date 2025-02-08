use bff::conformal::{
    boundary_edge_lengths, calc_angle_defects, calc_face_angles, cotan_laplacian_triplets,
    dirichlet_boundary, laplacian_set,
};
use bff::layout::{best_fit_curve, extend_curve};
use bff::mesh_structure::MeshStructure;
use bff::Result;
use faer::Mat;
use std::io::BufRead;
use std::path::Path;

fn main() -> Result<()> {
    let args = std::env::args().collect::<Vec<String>>();
    let default_file = Path::new("bumpcap.obj");

    let mesh = if args.len() == 1 && default_file.exists() {
        println!("No file specified, using default: {:?}", default_file);
        load_obj_file(default_file)
    } else if args.len() > 1 {
        let path = Path::new(&args[1]);
        load_obj_file(path)
    } else {
        println!("Usage: {} <path_to_obj_file>", args[0]);
        Err("Invalid arguments".into())
    }?;

    // The mesh structure was deciphered during the creation of the `MeshStructure` object,
    // which takes the vertices and faces and identifies the unique edges, the boundary loop(s),
    // and the edge/interior vertices. For more detail, see the `MeshStructure::new` function and
    // look at the code in the `mesh_structure.rs` module.

    // The rest of what follows is identical to the main test `end_to_end` in the `lib.rs` file,
    // which is specifically running on stored data from the `hyperboloid.obj` file from the Python
    // `confmap` (https://github.com/russelmann/confmap) implementation.
    let n_vert = mesh.vertices.len();

    // These are references to the &[u32] slices which define the indices of the interior and
    // boundary vertices (respectively) in the `mesh.vertices` list. They will be used in several
    // places below.
    let i_inner = mesh.inner_vertices()?;
    let i_bound = mesh.single_boundary_vertices()?;

    // Calculate the face angles, which will be used both to calculate the angle defects for the
    // dirichlet boundary condition and to calculate the cotangent weights for the laplacian.
    let face_angles = calc_face_angles(&mesh)?;

    // Calculate the cluster of values for the cotangent laplacian matrix and its sub-matrices.
    // We'll create the A matrix (the full cotangent laplacian) and the A_ii, A_ib, and A_bb
    // sub-matrices, which are the rows of the inner vertices against the inner vertices, the rows
    // of the inner vertices against the boundary vertices, and the rows of the boundary vertices
    // against the boundary vertices, respectively.
    let triplets = cotan_laplacian_triplets(&face_angles, n_vert, &mesh.edges, &mesh.face_edges)?;
    let (a, aii, aib, abb) = laplacian_set(n_vert, &i_inner, &i_bound, &triplets)?;

    // We'll pre-factor the A matrix and the A_ii matrix, the latter which will be used while
    // setting the dirichlet boundary condition and determining the positions of the boundary
    // vertices in the final layout, and the former which will be used to extend the outer layout
    // boundary vertices into the interior of the layout.
    let a_lu = a.sp_lu()?;
    let aii_lu = aii.sp_lu()?;

    // Calculate the angle defects to be used in the dirichlet boundary condition.
    let angle_defects = calc_angle_defects(n_vert, &i_bound, &face_angles, &mesh.faces)?;

    // The ub vector is set to zeroes when using the minimum distortion boundary condition.
    let ub = Mat::<f64>::zeros(i_bound.len(), 1);

    // Set the target im_k vector for the boundary vertices.
    let im_k = dirichlet_boundary(
        &ub,
        &aii_lu,
        &aib,
        &abb,
        &mesh.inner_vertices()?,
        &mesh.single_boundary_vertices()?,
        &angle_defects,
    )?;

    // Calculate the uv positions of the boundary vertices in the final layout
    let boundary_edge_len = boundary_edge_lengths(&mesh)?;
    let uvb = best_fit_curve(&ub, &im_k, &boundary_edge_len)?;

    // Finally, extend the boundary vertices into the interior of the layout.
    let uv = extend_curve(
        &a_lu,
        &aii_lu,
        &aib,
        &uvb,
        mesh.vertices.len(),
        &i_bound,
        &i_inner,
    )?;

    for row in uv.iter() {
        println!("{}, {}", row[0], row[1]);
    }

    Ok(())
}

/// A wavefront .obj file is a very simple text based format that describes the geometry of a 3D
/// model as a set of lines labeled with a single letter. To the extent that we care for this
/// program, lines starting with 'v' are vertices, lines starting with 'f' are faces, and lines
/// starting with # are comments.
///
/// This function will load a wavefront .obj file and return a MeshStructure.

fn load_obj_file(path: &Path) -> Result<MeshStructure> {
    let file = std::fs::File::open(path)?;
    let reader = std::io::BufReader::new(file);
    let mut vertices = Vec::new();
    let mut faces = Vec::new();

    for line in reader.lines() {
        let line = line?;
        let mut parts = line.split_whitespace();
        match parts.next() {
            Some("v") => {
                let x: f64 = parts.next().unwrap().parse()?;
                let y: f64 = parts.next().unwrap().parse()?;
                let z: f64 = parts.next().unwrap().parse()?;
                vertices.push([x, y, z]);
            }
            Some("f") => {
                let v1: u32 = parts.next().unwrap().split('/').next().unwrap().parse()?;
                let v2: u32 = parts.next().unwrap().split('/').next().unwrap().parse()?;
                let v3: u32 = parts.next().unwrap().split('/').next().unwrap().parse()?;
                faces.push([v1 - 1, v2 - 1, v3 - 1]);
            }
            _ => {}
        }
    }

    MeshStructure::new(vertices, faces)
}
