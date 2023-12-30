const std = @import("std");
const assert = std.debug.assert;

const g = @import("grad.zig");
const add = g.add;
const mul = g.mul;

const type_utils = @import("type_utils.zig");

fn Layer(comptime input_nodes: usize, comptime output_nodes: usize, comptime in_activation: fn (in: g.GradVal) g.GradVal) type {
    const layerType = struct {
        weights: [output_nodes][input_nodes]f64 = [_][input_nodes]f64{[_]f64{1} ** input_nodes} ** output_nodes,
        comptime param_count: usize = input_nodes * output_nodes,
        comptime input_size: usize = input_nodes,
        comptime output_size: usize = output_nodes,

        pub fn forward(weights: [output_nodes][input_nodes]f64, input: [input_nodes]f64) [output_nodes]f64 {
            var output: [output_nodes]f64 = [_]f64{0} ** output_nodes;
            for (0..output_nodes) |output_index| {
                var sum: f64 = 0;
                for (input, 0..) |input_val, input_index| {
                    sum += weights[output_index][input_index] * input_val;
                }
                sum /= input_nodes;
                output[output_index] = in_activation(g.GradVal{ .val = sum, .grad = 0 }).val;
            }
            return output;
        }

        //base_name used to ensure that each layer has uniquely named weights
        pub inline fn backward(weights: [output_nodes][input_nodes]f64, input: [input_nodes]g.GradVal, respect: u64, base_name: u64) [output_nodes]g.GradVal {
            var output: [output_nodes]g.GradVal = [_]g.GradVal{.{ .val = 0, .grad = 0 }} ** output_nodes;
            for (0..output_nodes) |output_index| {
                var sum: g.GradVal = g.literal(0);
                inline for (input, 0..) |input_val, input_index| {
                    //if it was a id list, here is a uniqid
                    var id = base_name + output_index * input_nodes + input_index;
                    sum = add(sum, mul(g.variable_uuid(weights[output_index][input_index], id, respect), input_val));
                }
                sum = g.div(sum, g.literal(input_nodes));

                //lack of inlining here is not as terrible as it could be, as zero terms are elsewhere in tree
                output[output_index] = in_activation(sum);
            }
            return output;
        }
    };
    return layerType;
}

//given a list of compiler defined layers:
//have a forward function that chains all of them
//have a backward function that does same with weights
//include gradient step based on an eval function -- could be in a dependent trainer class
fn NN(comptime layer_types: anytype, comptime inputs: usize, comptime outputs: usize) type {
    //type inference and instantiation is fun
    //need to turn []type -> { type1, type2.., etc. }

    //replace with typeof at some point, and see if there are issues with defaults
    const layers_flat_types = //@TypeOf(layer_types);
    type_utils.typeFlatten(layer_types);

    const nn_type = struct {
        layers: layers_flat_types = layers_flat_types{},

        //sigh I'll need to expose more I think
        // need to feed the output of each one into the next
        pub fn forward(layers: layers_flat_types, input: [inputs]f64) [outputs]f64 {
            var curr: []const f64 = input[0..];
            inline for (layer_types) |layer_type| {
                var dud: layer_type = layer_type{};
                assert(curr.len == dud.input_size);

                //need to understand how memory alloc works
                //disgusting, but now need to figure out how to do multiple layers properly
                curr = layer_type.forward(layers.l1.weights, @constCast(curr)[0..dud.input_size].*)[0..];
            }
            assert(curr.len == outputs);

            var out: [outputs]f64 = @constCast(curr)[0..outputs].*;
            //[_]f64{0} ** outputs;
            return out;
        }

        pub fn backward(layers: layers_flat_types, input: [inputs]g.GradVal, respect: u64) [outputs]g.GradVal {
            //base will be defined internally
            _ = respect;
            _ = input;
            _ = layers;
        }
    };

    return nn_type;
}

//overall goal is to get gradient of each weight with respect to the evaluation metric
//gradient of eval metric for a weight is calculated based on the gradient of previous
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

    try std.testing.expectEqual(l1.forward(layer.weights, .{1}), .{1});
    try std.testing.expectEqual(l1.forward(layer.weights, .{5}), .{5});
    try std.testing.expectEqual(l1.forward(layer.weights, .{-1}), .{0});
}

test "single layer gradient test" {
    const l1 = Layer(2, 1, g.relu);
    var layer: l1 = l1{};

    try std.testing.expectEqual(l1.forward(layer.weights, .{ 1, 1 }), .{1});
    try std.testing.expectEqual(l1.forward(layer.weights, .{ 1, -1 }), .{0});
    try std.testing.expectEqual(l1.forward(layer.weights, .{ 2, -1 }), .{0.5});
    try std.testing.expectEqual(l1.forward(layer.weights, .{ 1, -2 }), .{0});

    try std.testing.expectEqual(l1.backward(layer.weights, .{ g.literal(1), g.literal(1) }, 0, 0)[0].grad, 0.5);
    try std.testing.expectEqual(l1.backward(layer.weights, .{ g.literal(1), g.literal(1) }, 1, 0)[0].grad, 0.5);
}

test "network 2 layers forward" {
    //might allow a different input format
    const nn1 = NN(.{ Layer(1, 1, g.relu), Layer(1, 1, g.relu) }, 1, 1);

    var nn: nn1 = nn1{};
    //I literally can't access what it's evaluated to smh
    // std.debug.print("layers_flat_type: \n{}\n", .{@typeInfo(@TypeOf(nn.layers))});

    std.debug.print("\nType of nn:\n {}\n", .{@TypeOf(nn)});

    //
    try std.testing.expectEqual(nn1.forward(nn.layers, .{1}), .{1});
    // try std.testing.expectEqual(nn1.forward(nn.layers, .{5}), .{5});
    // try std.testing.expectEqual(nn1.forward(nn.layers, .{-1}), .{0});
}
