mod commands;
mod validation;
mod report;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: voxel-cli <command> [options]");
        eprintln!("Commands: generate, batch-test, inspect, sleep");
        std::process::exit(1);
    }

    match args[1].as_str() {
        "generate" => commands::generate::run(&args[2..]),
        "batch-test" => commands::batch_test::run(&args[2..]),
        "inspect" => commands::inspect::run(&args[2..]),
        "sleep" => commands::sleep::run(&args[2..]),
        _ => {
            eprintln!("Unknown command: {}", args[1]);
            std::process::exit(1);
        }
    }
}
