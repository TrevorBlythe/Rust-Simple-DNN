use Rust_Simple_DNN::rdnn::layers::*;
use Rust_Simple_DNN::rdnn::*;
// use Rust_Simple_DNN::rdnn::layers::GenericLayer::GenericLayer;

#[test]
fn test_relu_backward() {
    let input_data: Vec<f32> = vec![-1.0, 0.0, 1.0, -2.0, 2.0];
    let output_grads: Vec<f32> = vec![0.5, 1.0, 1.5, 2.0, 2.5];
    let expected_input_grads: Vec<f32> = vec![0.0, 1.0, 1.5, 0.0, 2.5];

    let mut layer = Relu::new(input_data.len());
    layer.forward_data(&input_data);
    layer.backward_grads(&input_data, &output_grads); // Updated to backward_grads
    let input_grads = layer.get_input_grads().clone(); // Updated to get_input_grads

    assert_eq!(input_grads, expected_input_grads, "ReLU backward failed");
}

#[test]
fn test_sigmoid_backward() {
    let input_data: Vec<f32> = vec![-1.0, 0.0, 1.0];
    let output_grads: Vec<f32> = vec![0.5, 1.0, 1.5];

    let mut layer = Sigmoid::new(input_data.len());
    layer.forward_data(&input_data);
    layer.backward_grads(&input_data, &output_grads);
    let input_grads = layer.get_input_grads().clone();

    let expected_input_grads: Vec<f32> = vec![0.09830597, 0.25, 0.29491788]; //Corrected expected values

    for (grad, expected) in input_grads.iter().zip(expected_input_grads.iter()) {
        assert!((grad - expected).abs() < 1e-6, "Sigmoid backward failed: {:?}", grad);
    }
}
#[test]
fn test_tanh_backward() {
    let input_data: Vec<f32> = vec![-1.0, 0.0, 1.0];
    let output_grads: Vec<f32> = vec![0.5, 1.0, 1.5];

    let mut layer = Tanh::new(input_data.len());
    layer.forward_data(&input_data);
    layer.backward_grads(&input_data, &output_grads); // Updated to backward_grads
    let input_grads = layer.get_input_grads().clone(); // Updated to get_input_grads

    let expected_input_grads: Vec<f32> = vec![0.20998716, 1.0, 0.6299615];

    for (grad, expected) in input_grads.iter().zip(expected_input_grads.iter()) {
        assert!((grad - expected).abs() < 1e-6, "Tanh backward failed: {:?}", grad);
    }
}

#[test]
fn test_relu_learn_positive_values() {
    let mut net = Net::new(
        vec![Relu::new(5)], // Simple ReLU layer
        1,
        0.01,
    );

    let mut training_iteration_counter = 0;
    while training_iteration_counter < 10000 {
        let input_positive: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let output_positive = net.forward_data(&input_positive).clone();
        net.backward_data(&vec![1.0, 2.0, 3.0, 4.0, 5.0]); // Target: same as input

        let input_negative: Vec<f32> = vec![-1.0, -2.0, -3.0, -4.0, -5.0];
        let output_negative = net.forward_data(&input_negative).clone();
        net.backward_data(&vec![0.0, 0.0, 0.0, 0.0, 0.0]); // Target: all zeros

        if training_iteration_counter % 1000 == 0 {
            println!(
                "Iteration {}: Positive Output: {:?}, Negative Output: {:?}",
                training_iteration_counter, output_positive, output_negative
            );
        }

        training_iteration_counter += 1;
    }

    // Test after training
    let test_positive: Vec<f32> = vec![6.0, 7.0, 8.0, 9.0, 10.0];
    let output_positive_test = net.forward_data(&test_positive).clone();

    let test_negative: Vec<f32> = vec![-6.0, -7.0, -8.0, -9.0, -10.0];
    let output_negative_test = net.forward_data(&test_negative);

    for (out, expected) in output_positive_test.iter().zip(test_positive.iter()) {
        assert!((out - expected).abs() < 0.1, "ReLU positive test failed: {:?}", out);
    }

    for out in output_negative_test.iter() {
        assert!(out.abs() < 0.1, "ReLU negative test failed: {:?}", out);
    }
}

#[test]
fn test_sigmoid_learn_binary_classification() {
    let mut net = Net::new(
        vec![Sigmoid::new(3)], // Simple Sigmoid layer
        1,
        0.1,
    );

    let mut training_iteration_counter = 0;
    while training_iteration_counter < 10000 {
        let input_positive: Vec<f32> = vec![1.0, 2.0, 3.0];
        let output_positive = net.forward_data(&input_positive).clone();
        net.backward_data(&vec![1.0, 1.0, 1.0]); // Target: 1.0 (positive)

        let input_negative: Vec<f32> = vec![-1.0, -2.0, -3.0];
        let output_negative = net.forward_data(&input_negative).clone();
        net.backward_data(&vec![0.0, 0.0, 0.0]); // Target: 0.0 (negative)

        if training_iteration_counter % 1000 == 0 {
            println!(
                "Iteration {}: Positive Output: {:?}, Negative Output: {:?}",
                training_iteration_counter, output_positive, output_negative
            );
        }

        training_iteration_counter += 1;
    }

    // Test after training
    let test_positive: Vec<f32> = vec![4.0, 5.0, 6.0];
    let output_positive_test = net.forward_data(&test_positive).clone();

    let test_negative: Vec<f32> = vec![-4.0, -5.0, -6.0];
    let output_negative_test = net.forward_data(&test_negative);

    for out in output_positive_test.iter() {
        assert!(out > &0.8, "Sigmoid positive test failed: {:?}", out);
    }

    for out in output_negative_test.iter() {
        assert!(out < &0.2, "Sigmoid negative test failed: {:?}", out);
    }
}

#[test]
fn test_tanh_learn_range() {
    let mut net = Net::new(
        vec![Tanh::new(3)], // Simple Tanh layer
        1,
        0.01,
    );

    let mut training_iteration_counter = 0;
    while training_iteration_counter < 10000 {
        let input_positive: Vec<f32> = vec![1.0, 2.0, 3.0];
        let output_positive = net.forward_data(&input_positive).clone();
        net.backward_data(&vec![1.0, 1.0, 1.0]); // Target: 1.0

        let input_negative: Vec<f32> = vec![-1.0, -2.0, -3.0];
        let output_negative = net.forward_data(&input_negative).clone();
        net.backward_data(&vec![-1.0, -1.0, -1.0]); // Target: -1.0

        if training_iteration_counter % 1000 == 0 {
            println!(
                "Iteration {}: Positive Output: {:?}, Negative Output: {:?}",
                training_iteration_counter, output_positive, output_negative
            );
        }

        training_iteration_counter += 1;
    }

    // Test after training
    let test_positive: Vec<f32> = vec![4.0, 5.0, 6.0];
    let output_positive_test = net.forward_data(&test_positive).clone();

    let test_negative: Vec<f32> = vec![-4.0, -5.0, -6.0];
    let output_negative_test = net.forward_data(&test_negative);

    for out in output_positive_test.iter() {
        assert!(out > &0.8, "Tanh positive test failed: {:?}", out);
    }

    for out in output_negative_test.iter() {
        assert!(out < &-0.8, "Tanh negative test failed: {:?}", out);
    }
}