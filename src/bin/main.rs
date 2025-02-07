use std::path::Path;
use bff::mesh_structure::MeshStructure;
use bff::Result;
use bff::serialize::MeshData;

fn main() -> Result<()> {
    let target = Path::new("../sample_data/hyperboloid.msgpack");
    println!("Loading mesh data from {:?}", target);

    let mesh = MeshData::load(&target)?;
    let structure = MeshStructure::new(mesh.vertices, mesh.faces);


    Ok(())
}
