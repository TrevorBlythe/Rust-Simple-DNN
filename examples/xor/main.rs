use Rust_Simple_DNN::rdnn::layers::*;
use Rust_Simple_DNN::rdnn::*;


fn main() {
	
	//example model
	let mut net = Net::new(
        vec![
            FC::new(2, 4), //input size 2, output size 4
            Sig::new(4), //sigmoid, input size 4 output size 4

            FC::new(4, 4),
            Sig::new(4), //sigmoid

            FC::new(4, 1),// input 4 output 1
            Sig::new(1), //sigmoid
        ],
        1, //batch size
        0.1, //learning rate
    );

    let mut x = 0;

    //training loop
    while x < 5000 {
        net.forward_data(&vec![1.0, 0.0]);
        net.backward_data(&vec![1.0]);

        net.forward_data(&vec![1.0, 1.0]);
        net.backward_data(&vec![0.0]);

        net.forward_data(&vec![0.0, 1.0]);
        net.backward_data(&vec![1.0]);

        net.forward_data(&vec![0.0, 0.0]);
        net.backward_data(&vec![0.0]);
        x += 1;
    }

    println!("Output from network:");
    println!("{:?}", net.forward_data(&vec![1.0, 0.0]));
    println!("{:?}", net.forward_data(&vec![0.0, 0.0]));

}