use std::path::Path;
use bff::Result;
use bff::serialize::MeshData;

fn main() -> Result<()> {
    let target = Path::new("../sample_data/hyperboloid.msgpack");
    println!("Loading mesh data from {:?}", target);
    let mesh = MeshData::load(&target)?;

    println!(" * Vertices: {}", mesh.vertices.len());
    println!(" * Faces: {}", mesh.faces.len());

    Ok(())
}
