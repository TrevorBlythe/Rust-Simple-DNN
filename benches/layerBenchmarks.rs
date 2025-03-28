use Rust_Simple_DNN::rdnn::layers::*;
use Rust_Simple_DNN::rdnn::layers::GenericLayer::GenericLayer;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
 // Assuming your layers are in this module

fn bench_relu_forward(c: &mut Criterion) {
    let mut layer = Relu::new(1024);
    let input_data: Vec<f32> = vec![1.0; 1024];

    c.bench_function("ReLU Forward", |b| {
        b.iter(|| layer.forward_data(black_box(&input_data)))
    });
}

fn bench_relu_backward_grads(c: &mut Criterion) {
    let mut layer = Relu::new(1024);
    let input_data: Vec<f32> = vec![1.0; 1024];
    let grads: Vec<f32> = vec![1.0; 1024];
    layer.forward_data(&input_data); // Need forward pass first

    c.bench_function("ReLU Backward Grads", |b| {
        b.iter(|| layer.backward_grads(black_box(&input_data), black_box(&grads)))
    });
}

fn bench_leaky_relu_forward(c: &mut Criterion) {
    let mut layer = LeakyRelu::new(1024,0.01);
    let input_data: Vec<f32> = vec![1.0; 1024];

    c.bench_function("Leaky ReLU Forward", |b| {
        b.iter(|| layer.forward_data(black_box(&input_data)))
    });
}

fn bench_leaky_relu_backward_grads(c: &mut Criterion) {
    let mut layer = LeakyRelu::new(1024,0.01);
    let input_data: Vec<f32> = vec![1.0; 1024];
    let grads: Vec<f32> = vec![1.0; 1024];
    layer.forward_data(&input_data); // Need forward pass first

    c.bench_function("Leaky ReLU Backward Grads", |b| {
        b.iter(|| layer.backward_grads(black_box(&input_data), black_box(&grads)))
    });
}

fn bench_sigmoid_forward(c: &mut Criterion) {
    let mut layer = Sigmoid::new(1024);
    let input_data: Vec<f32> = vec![1.0; 1024];

    c.bench_function("Sigmoid Forward", |b| {
        b.iter(|| layer.forward_data(black_box(&input_data)))
    });
}

fn bench_sigmoid_backward_grads(c: &mut Criterion) {
    let mut layer = Sigmoid::new(1024);
    let input_data: Vec<f32> = vec![1.0; 1024];
    let grads: Vec<f32> = vec![1.0; 1024];
    layer.forward_data(&input_data); // Need forward pass first

    c.bench_function("Sigmoid Backward Grads", |b| {
        b.iter(|| layer.backward_grads(black_box(&input_data), black_box(&grads)))
    });
}

fn bench_tanh_forward(c: &mut Criterion) {
    let mut layer = Tanh::new(1024);
    let input_data: Vec<f32> = vec![1.0; 1024];

    c.bench_function("Tanh Forward", |b| {
        b.iter(|| layer.forward_data(black_box(&input_data)))
    });
}

fn bench_tanh_backward_grads(c: &mut Criterion) {
    let mut layer = Tanh::new(1024);
    let input_data: Vec<f32> = vec![1.0; 1024];
    let grads: Vec<f32> = vec![1.0; 1024];
    layer.forward_data(&input_data); // Need forward pass first

    c.bench_function("Tanh Backward Grads", |b| {
        b.iter(|| layer.backward_grads(black_box(&input_data), black_box(&grads)))
    });
}

criterion_group!(
    benches,
    bench_relu_forward,
    bench_relu_backward_grads,
    bench_leaky_relu_forward,
    bench_leaky_relu_backward_grads,
    bench_sigmoid_forward,
    bench_sigmoid_backward_grads,
    bench_tanh_forward,
    bench_tanh_backward_grads
);
criterion_main!(benches);