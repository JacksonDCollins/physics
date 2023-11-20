use std::{collections::HashMap, env, path::Path};

use glsl_to_spirv::ShaderType;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=shaders");

    let out_dir = get_output_path(); //std::env::var("OUT_DIR").unwrap();
                                     // Create destination path if necessary
    std::fs::create_dir_all("shaders")?;
    std::fs::create_dir_all(format!("{}/shaders", out_dir))?;

    let mut generated_files: HashMap<String, String> = HashMap::new();

    for entry in std::fs::read_dir("shaders")? {
        let entry = entry?;

        if entry.file_type()?.is_file() {
            let in_path = entry.path();

            // Support only vertex and fragment shaders currently
            let shader_type =
                in_path
                    .extension()
                    .and_then(|ext| match ext.to_string_lossy().as_ref() {
                        "glsl" => match in_path
                            .file_stem()
                            .and_then(|stem| stem.to_str().and_then(|s| s.split('_').last()))
                        {
                            Some(shader_type) => match shader_type {
                                "vert" => Some(ShaderType::Vertex),
                                "frag" => Some(ShaderType::Fragment),
                                "compute" => Some(ShaderType::Compute),
                                _ => None,
                            },
                            _ => None,
                        },
                        _ => None,
                    });

            if let Some(shader_type) = shader_type {
                use std::io::Read;

                let source = std::fs::read_to_string(&in_path)?;
                let mut compiled_file = glsl_to_spirv::compile(&source, shader_type)?;

                // Read the binary data from the compiled file
                let mut compiled_bytes = Vec::new();
                compiled_file.read_to_end(&mut compiled_bytes)?;

                // Determine the output path based on the input name
                let out_path = format!(
                    "{}/shaders/{}.spv",
                    out_dir,
                    in_path.file_stem().unwrap().to_string_lossy()
                );

                std::fs::write(&out_path, &compiled_bytes)?;

                generated_files.insert(
                    in_path.file_stem().unwrap().to_string_lossy().to_string(),
                    out_path,
                );
            }
        }
    }

    std::fs::write("shaders/out_file.json", format!("{:?}", generated_files))?;

    Ok(())
}

fn get_output_path() -> String {
    //<root or manifest path>/target/<profile>/
    let manifest_dir_string = env::var("CARGO_MANIFEST_DIR").unwrap();
    let build_type = env::var("PROFILE").unwrap();
    let path = Path::new(&manifest_dir_string)
        .join("target")
        .join(build_type);
    path.to_string_lossy().to_string()
}
