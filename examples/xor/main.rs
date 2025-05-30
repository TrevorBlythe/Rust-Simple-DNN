use Rust_Simple_DNN::rdnn::layers::*;
use Rust_Simple_DNN::rdnn::*;


fn main() {
	
	//example model
	let mut net = Net::new(
        vec![
            FC::new(2, 3), //input size 2, output size 4
            LeakyRelu::new(3,0.1), //LeakyRelu, input size 4 output size 4

            FC::new(3, 3),
            LeakyRelu::new(3,0.1), //LeakyRelu

            FC::new(3, 1),// input 4 output 1
            LeakyRelu::new(1,0.1), //LeakyRelu
        ],
        1, //batch size
        0.1, //learning rate, this is really high but for something simple its fine
    );

    let mut training_iteration_counter = 0;

    //training loop
    while training_iteration_counter < 1500 {
        net.forward_data(&vec![1.0, 0.0]);
        net.backward_data(&vec![1.0]);

        net.forward_data(&vec![1.0, 1.0]);
        net.backward_data(&vec![0.0]);

        net.forward_data(&vec![0.0, 1.0]);
        net.backward_data(&vec![1.0]);

        net.forward_data(&vec![0.0, 0.0]);
        net.backward_data(&vec![0.0]);
        training_iteration_counter += 1;
    }

    println!("Output from network (should be 1, 0):");
    println!("{:?}", net.forward_data(&vec![1.0, 0.0]));
    println!("{:?}", net.forward_data(&vec![0.0, 0.0]));

}