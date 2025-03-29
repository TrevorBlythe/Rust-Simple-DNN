

# Rust DNN

Create Modular Lightweight Deep Neural Networks in Rust easy

# Installation

After running

```
cargo add Rust_Simple_DNN
```

Then you must put these in your rust code

```rust
use Rust_Simple_DNN::rdnn::layers::*;
use Rust_Simple_DNN::rdnn::*;
```

# Current Implemented Layers

Think of layers as building blocks for a neural network. Different Layers process data in different ways. Layers can be trained.
### layers:

- Fully connected Dense Layers

```rust
FC::new(inputSize, outputSize)
```

These are best when doing just straight raw brain processing. Using these combined with activations, it is technically possible to make a mathematical ai for anything.
These layers have exponintial more computation when scaled up though.

<br>

- Activations

```rust
Tanh::new(inputSize); //hyperbolic tangent
Relu::new(inputSize); //if activation > 0
LeakyRelu::new(inputSize, alpha); //Relu with leak
Sigmoid::new(inputSize); //sigmoid
```

<br>

- Convolutions

```rust
Conv2d::new(input_dims: [usize; 3],
        filter_dims: [usize; 2],
        output_channels: usize, //also represents the amount of filters
        stride: usize,);
```

# starting tutorial

This is how you make a neural network that looks like this
<br>
<img src="network.png" alt="image-alt-text-check-github-to-see-image" width="300"/>

Use this code to make it:

```rust

//Model/network/AI Definition
let mut net = Net::new(
        vec![
            FC::new(3, 4), //Linear/Dense input size 3, output 4
            Sigmoid::new(4), //sigmoid, input 4 output 4

            FC::new(4, 4),
            Sigmoid::new(4), //sigmoid

            FC::new(4, 1),// input 4 output 1
            Sigmoid::new(1), //sigmoid
        ],
        1, //batch size
        0.1, //learning rate
    );
```

<br>
<br>
This is how you *propagate data* through the network:

```rust
net.forward_data(&vec![1.0, 0.0, -69.0]); //returns the output vector from the Model
```

After propagating data through, you can then backpropagate your target:

```rust
// This parameter is the models target, (aka what you want the ai to output)
 net.backward_data(&vec![0.0]); //trains the ai to output 0
```

The network will store and apply the gradients, so to train the network, all you need to do is repeatedly forward and back-propagate your data in order

```rust
//TRAINING LOOP

let mut iteration = 0; //just a counter
    while iteration < 5000 {
        net.forward_data(&vec![1.0, 0.0, 0.0]);
        net.backward_data(&vec![1.0]);

        net.forward_data(&vec![1.0, 1.0, 0.0]);
        net.backward_data(&vec![0.0]);

        net.forward_data(&vec![0.0, 1.0, 0.0]);
        net.backward_data(&vec![1.0]);

        net.forward_data(&vec![0.0, 0.0, 0.0]);
        net.backward_data(&vec![0.0]);
        iteration += 1;
    }

//at this point its well trained
```

# Saving and Loading

An example for saving in loading is in examples/saveLoadModel.
You have to be using the serde feature on this crate
```toml
[dependencies]
Rust_Simple_DNN = { ... features= ["serde"]}
#use this feature to save and load to files
```

In code, the functions for saving and loading are:
```rust
//Some example model you want to save..
let mut net = Net::new(
    ... //dont forget the architecture it doesn't get saved
);
net.save_weights("path.json"); //Stores the parameters


let mut loaded_net = Net::new( 
    ... //same architecture as "net" or it wont load right
);
loaded_net.load_weights("path.json"); //loaded net will now produce the same output as net
```
