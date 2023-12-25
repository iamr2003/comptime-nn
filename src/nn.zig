
const std = @import("std");

const g = @import("grad.zig");

//I might not need to be quite this silly
//might just implement the autograd way and use zig comptime to make it happen
//let's propogate through a single layer first
fn Layer(comptime input_nodes: usize, comptime output_nodes: usize, comptime in_activation: fn (in: g.GradVal) g.GradVal) type {
    const layerType = struct {
        weights: [output_nodes][input_nodes]f64 = [_][input_nodes]f64{[_]f64{1} ** input_nodes} ** output_nodes,
        comptime input_size: usize = input_nodes,
        comptime output_size: usize = output_nodes,
        
        pub fn forward(weights: [output_nodes][input_nodes]f64, input: [input_nodes]f64) [output_nodes]f64 {
            //will need to be converted into a gradval safe situation
            var output: [output_nodes]f64 = [_]f64{0} ** output_nodes;
            for (0..output_nodes) |output_index| {
                var sum: f64 = 0;
                for (input, 0..) |input_val, input_index| {
                    sum += weights[output_index][input_index] * input_val;
                }
                sum /= input_nodes;
                //yes inefficiency, will recompute with gradients
                output[output_index] = in_activation(g.GradVal{.val = sum, .grad = 0}).val;
            }
            return output;
        }

        //goal is to return each gradient of each output with respect to each weight
        //which then needs to be chained with eval metric
        //yep we'll still go without simdeez for rn
        
    };
    return layerType;
}

// const nonCompLayer = struct {
//     //some annoying stuff with partial initialization
//     weights: [][]f64,
//     input_size: usize,
//     output_size: usize,
//     activation: fn (in: f64) f64,
//
//     pub fn init(comptime input: usize, comptime output: usize, comptime act: fn (in: f64) f64) nonCompLayer {
//         return nonCompLayer{ .weights = [_][input]f64{[_]f64{1} ** input} ** output, .in_activation = act, .input_size = input, .output_size = output };
//     }
//
//     pub fn forward(l: nonCompLayer, input: []f64) []f64 {
//         var output: [l.input_size]f64 = [_]f64{0} ** l.output_size;
//         for (0..l.output_size) |output_index| {
//             var sum: f64 = 0;
//             for (input, 0..) |input_val, input_index| {
//                 sum += l.weights[output_index][input_index] * input_val;
//             }
//             sum /= l.input_size;
//             output[output_index] = l.activation(sum);
//         }
//         return output;
//     }
// };
//
// const nonCompNetwork = struct {
//     layers: []nonCompLayer,
//
//     pub fn forward(n: nonCompNetwork, input: []f64) []f64 {
//         for (n.layers) |l| {
//             input = l.forward(input);
//         }
//         return input;
//     }
// };
//
// fn Network(comptime input_layers: []const type) type {
//     comptime {
//         //initialize one of each type to get info
//         const front: input_layers[0] = input_layers[0]{};
//         const back: input_layers[input_layers.len - 1] = input_layers[input_layers.len - 1]{};
//
//         const network_type = struct {
//             //need to figure out
//             comptime layers: []const type = input_layers,
//             //no I need to store values as well
//             // // hardcode dumbness, also annoyances of functions being from type
//             // // and not member
//             pub fn forward(layers: []const type, in: [front.input_size]f64) [back.output_size]f64 {
//                 //messiness should hopefully get cleaned up by compiler
//                 //would like to do this recursively tbh, composition of functions
//                 // need to compose a list of functions
//
//                 return layers[0].forward(front[0].weights, in);
//             }
//         };
//         return network_type;
//     }
// }

inline fn relu(in: f64) f64 {
    return @max(0, in);
}

pub fn main() anyerror!void {
    // _ = nn;
    std.log.info("All your codebase are belong to us.", .{});
}

test "layer test single var" {
    const l1 = Layer(1, 1, g.relu);
    var layer: l1 = l1{};

    // var nn: nn_type = .{ .layers = @TypeOf(nn_type.layers).init() };

    try std.testing.expectEqual(l1.forward(layer.weights, .{1}), .{1});
    try std.testing.expectEqual(l1.forward(layer.weights, .{5}), .{5});
    try std.testing.expectEqual(l1.forward(layer.weights, .{-1}), .{0});
}

// test "non comp layer" {
//     const l: nonCompLayer = nonCompLayer.init(1, 1, relu);
//     try std.testing.expectEqual(nonCompLayer.forward(l, .{1}), .{1});
//     try std.testing.expectEqual(nonCompLayer.forward(l, .{-1}), .{0});
//     try std.testing.expectEqual(nonCompLayer.forward(l, .{5}), .{5});
// }

// test "nn test single var single layer" {
//     const layers: [1]type = .{Layer(1, 1, relu)};
//     const nn_type = Network(&layers);
//     var nn: nn_type = nn_type{};
//     // //function not preserved?
//     try std.testing.expectEqual(nn_type.forward(nn.layer, .{-1}), .{0});
//     try std.testing.expectEqual(nn_type.forward(nn.layer, .{1}), .{1});
// }

// test "array of types" {
//     comptime{
//         const arr_types:[3]type = .{i64,f64,i32};
//         _ = arr_types;
//     }
// }
