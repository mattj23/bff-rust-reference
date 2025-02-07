use bff::mesh_structure::MeshStructure;
use bff::serialize::MeshData;
use bff::Result;
use std::path::Path;

fn main() -> Result<()> {
    let target = Path::new("../sample_data/hyperboloid.msgpack");
    println!("Loading mesh data from {:?}", target);

    let mesh = MeshData::load(&target)?;
    let structure = MeshStructure::new(mesh.vertices, mesh.faces);

    Ok(())
}
