use std::{collections::HashMap, env, path::Path};

// use glsl_to_spirv::ShaderType;
use shaderc;

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

            let shader_type =
                in_path
                    .extension()
                    .and_then(|ext| match ext.to_string_lossy().as_ref() {
                        "glsl" => match in_path
                            .file_stem()
                            .and_then(|stem| stem.to_str().and_then(|s| s.split('_').last()))
                        {
                            Some(shader_type) => match shader_type {
                                "vert" => Some(shaderc::ShaderKind::Vertex),
                                "frag" => Some(shaderc::ShaderKind::Fragment),
                                "compute" => Some(shaderc::ShaderKind::Compute),
                                _ => None,
                            },
                            _ => None,
                        },
                        _ => None,
                    });

            if let Some(shader_type) = shader_type {
                let source = std::fs::read_to_string(&in_path)?;

                let compiler = shaderc::Compiler::new().unwrap();
                let options = shaderc::CompileOptions::new().unwrap();

                let binary_result = compiler.compile_into_spirv(
                    &source,
                    shader_type,
                    in_path.to_str().unwrap(),
                    "main",
                    Some(&options),
                )?;

                // Determine the output path based on the input name
                let out_path = format!(
                    "{}/shaders/{}.spv",
                    out_dir,
                    in_path.file_stem().unwrap().to_string_lossy()
                );

                std::fs::write(&out_path, binary_result.as_binary_u8())?;

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
