#![allow(non_snake_case)]

pub mod FC;
pub mod GenericLayerTrait;
pub use GenericLayerTrait::GenericLayer; // Re-export the trait
pub mod Identity;
pub mod Sigmoid;
pub mod Conv2D;
pub mod Relu;
pub mod Tanh;
pub mod LeakyRelu;