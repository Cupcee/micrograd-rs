use plotters::prelude::*;
use std::{fs, iter::zip};

pub fn draw_chart(x: &Vec<(f32, f32)>, y01: &Vec<f32>) -> Result<(), Box<dyn std::error::Error>> {
    fs::create_dir("plots")?;
    let root = BitMapBackend::new("plots/test.png", (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;

    let gradient = colorous::VIRIDIS;
    let mut chart = ChartBuilder::on(&root)
        .caption("moons", ("sans-serif", 24).into_font())
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(-2f32..2f32, -2f32..2f32)?;

    chart.configure_mesh().draw()?;

    let xy = zip(x.clone(), y01);
    chart
        .draw_series(xy.into_iter().map(|(x, y)| {
            let color = gradient.eval_continuous((*y).into());
            Circle::new(x, 3, RGBColor(color.r, color.g, color.b).filled())
        }))
        .unwrap();

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;

    Ok(())
}
