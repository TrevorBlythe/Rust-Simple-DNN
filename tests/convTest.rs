use Rust_Simple_DNN::rdnn::layers::*;
use Rust_Simple_DNN::rdnn::*;


#[test]
fn test_weird_dimensions() {
    let mut net = Net::new(
        vec![Conv2D::new([3, 3, 1], [2, 2], 1, 2)],
        1,
        0.5,
    );

    let input_data: Vec<f32> = vec![
        1.0, 0.0, 1.0,
        0.0, 1.0, 0.0, 
        1.0, 0.0, 1.0,
    ];

    let mut mutable_params = net.layers[1].get_params_mut(); // Corrected index to 0
    mutable_params[0].clone_from_slice(&vec![
        0.0, 1.0,
        1.0, 0.0,
    ]);
    mutable_params[1].fill(0.0);
    let output = net.forward_data(&input_data);

    let expected_output: Vec<f32> = vec![
        0.0,
    ];
    
    assert_eq!(output, &expected_output, "Output was: {:?}", output);
}

#[test]
fn test_weird_dimensions2() {
    let mut net = Net::new(
        vec![Conv2D::new([4, 4, 1], [2, 2], 1, 2)],
        1,
        0.5,
    );

    let input_data: Vec<f32> = vec![
        1.0, 0.0, 1.0, 0.0,
        0.0, 1.0, 0.0, 1.0,
        1.0, 0.0, 1.0, 0.0,
        0.0, 1.0, 0.0, 1.0,

    ];

    let mut mutable_params = net.layers[1].get_params_mut(); // Corrected index to 0
    mutable_params[0].clone_from_slice(&vec![
        0.0, 1.0,
        1.0, 0.0,
    ]);
    mutable_params[1].fill(0.0);
    let output = net.forward_data(&input_data);

    let expected_output: Vec<f32> = vec![
        0.0,0.0,
        0.0,0.0,
    ];
    
    assert_eq!(output, &expected_output, "Output was: {:?}", output);
}


#[test]
fn test_conv2d_forward_stride_2_two_filters_with_stride() {
    let mut net = Net::new(
        vec![Conv2D::new([5, 5, 1], [3, 3], 2, 2)],
        1,
        0.5,
    );

    let input_data: Vec<f32> = vec![
        1.0, 0.0, 1.0, 0.0, 1.0,
        0.0, 1.0, 0.0, 1.0, 0.0,
        1.0, 0.0, 1.0, 0.0, 1.0,
        0.0, 1.0, 0.0, 1.0, 0.0,
        1.0, 0.0, 1.0, 0.0, 1.0,
    ];

    let mut mutable_params = net.layers[1].get_params_mut(); // Corrected index to 0
    mutable_params[0].clone_from_slice(&vec![
        0.0, 1.0, 0.0,
        1.0, 0.0, 1.0, //filter 1
        0.0, 1.0, 0.0,

        1.0, 0.0, 1.0,
        0.0, 1.0, 0.0, //filter 2
        1.0, 0.0, 1.0,
    ]);
    mutable_params[1].fill(0.0);

    let output = net.forward_data(&input_data);

    let expected_output: Vec<f32> = vec![
        0.0, 0.0,
        0.0, 0.0,

        5.0, 5.0,
        5.0, 5.0,
    ];

    assert_eq!(output, &expected_output, "Output was: {:?}", output);
}

#[test]
fn test_conv2d_forward_stride_2_two_filters_with_stride_and_input_depth() {
    let mut net = Net::new(
        vec![Conv2D::new([5, 5, 2], [3, 3], 3, 2)],
        1,
        0.5,
    );

    let input_data: Vec<f32> = vec![
        1.0, 0.0, 1.0, 0.0, 1.0,
        0.0, 1.0, 0.0, 1.0, 0.0,
        1.0, 0.0, 1.0, 0.0, 1.0,
        0.0, 1.0, 0.0, 1.0, 0.0,
        1.0, 0.0, 1.0, 0.0, 1.0,


        0.0, 1.0, 0.0, 1.0, 0.0,
        1.0, 0.0, 1.0, 0.0, 1.0,
        0.0, 1.0, 0.0, 1.0, 0.0,
        1.0, 0.0, 1.0, 0.0, 1.0,
        0.0, 1.0, 0.0, 1.0, 0.0,
    ];

    let mut mutable_params = net.layers[1].get_params_mut(); // Corrected index to 0
    mutable_params[0].clone_from_slice(&vec![
        0.0, 1.0, 0.0,
        1.0, 0.0, 1.0, 
        0.0, 1.0, 0.0,
                        //filter 1 (doesn't align with the pattern at all)
        1.0, 0.0, 1.0,
        0.0, 1.0, 0.0, 
        1.0, 0.0, 1.0,

        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 
        0.0, 0.0, 0.0,
                        //filter 2 
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 
        0.0, 0.0, 0.0,


        1.0, 0.0, 1.0,
        0.0, 1.0, 0.0, 
        1.0, 0.0, 1.0,
                        //filter 3 (aligns)
        0.0, 1.0, 0.0,
        1.0, 0.0, 1.0, 
        0.0, 1.0, 0.0,
                        


    ]);
    mutable_params[1].fill(0.0);

    let output = net.forward_data(&input_data);

    let expected_output: Vec<f32> = vec![
        0.0, 0.0,
        0.0, 0.0,

        0.0, 0.0,
        0.0, 0.0,

        9.0, 9.0,
        9.0, 9.0,
    ];

    assert_eq!(output, &expected_output, "Output was: {:?}", output);
}


#[test]
fn test_conv2d_forward_basic() {
    let mut net = Net::new(
        vec![Conv2D::new([6, 6, 1], [3, 3], 1, 1)],
        1,
        0.5,
    );

    let input_data: Vec<f32> = vec![
        0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
        0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
        0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
        0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
        0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
        0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
    ];

    let mut mutable_params = net.layers[1].get_params_mut();
    mutable_params[0].clone_from_slice(&vec![
        0.0, 1.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 1.0, 0.0,
    ]);
    mutable_params[1].fill(-1.0);

    let output = net.forward_data(&input_data);

    let expected_output: Vec<f32> = vec![
        2.0, -1.0, 2.0, -1.0,
        2.0, -1.0, 2.0, -1.0,
        2.0, -1.0, 2.0, -1.0,
        2.0, -1.0, 2.0, -1.0,
    ];

    assert_eq!(output, &expected_output, "Output was: {:?}", output);
}



#[test]
fn test_conv2d_learn_vertical_lines() {
    let mut net = Net::new(
        vec![Conv2D::new([5, 5, 1], [3, 3], 1, 1)], // Detect vertical lines
        1,
        0.01,
    );

    //remove bias so we can focus on if the weights are training right
    let mut mutable_params = net.layers[1].get_params_mut();
    mutable_params[1].fill(-1.0);

    let mut training_iteration_counter = 0;
    while training_iteration_counter < 10000 {
        // Vertical line pattern
        let input_vertical_line: Vec<f32> = vec![
            1.0, 0.0, 1.0, 0.0, 1.0,
            1.0, 0.0, 1.0, 0.0, 1.0,
            1.0, 0.0, 1.0, 0.0, 1.0,
            1.0, 0.0, 1.0, 0.0, 1.0,
            1.0, 0.0, 1.0, 0.0, 1.0,
        ];
        let output_for_input_with_line = net.forward_data(&input_vertical_line).clone();
        net.backward_data(
        &vec![
        1.0,0.0,1.0,
        1.0,0.0,1.0,
        1.0,0.0,1.0,
        ]); // Target: 1.0 (line detected)

        // No vertical line pattern
        let input_no_line: Vec<f32> = vec![
            0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let output_for_input_without_line = net.forward_data(&input_no_line).clone();
        net.backward_data(
        &vec![
            0.0,0.0,0.0,
            0.0,0.0,0.0,
            0.0,0.0,0.0,
        ]); // Target: 0.0 (no line)

        if training_iteration_counter % 100 == 0 {
            println!(
                "Iteration {}: Line Output: {:?}, No Line Output: {:?}",
                training_iteration_counter, output_for_input_with_line, output_for_input_without_line
            );
        }

        training_iteration_counter += 1;
    }

    // Test after training
    let vertical_line_test: Vec<f32> = vec![
        1.0, 0.0, 1.0, 0.0, 1.0,
        1.0, 0.0, 1.0, 0.0, 1.0,
        1.0, 0.0, 1.0, 0.0, 1.0,
        1.0, 0.0, 1.0, 0.0, 1.0,
        1.0, 0.0, 1.0, 0.0, 1.0,
    ];
    let output_vertical = net.forward_data(&vertical_line_test)[0];

    let no_line_test: Vec<f32> = vec![
        0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0,
    ];
    let output_no_line_vec = net.forward_data(&no_line_test);
    let output_no_line = output_no_line_vec.iter().fold(0.0, |max:f32, &val| max.max(val));

    assert!(output_vertical > 0.8, "Vertical line not detected: {:?}", output_vertical);
    assert!(output_no_line < 0.2, "No false positive");
}