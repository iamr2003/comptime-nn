const std = @import("std");
const rand = std.rand;
const nn = @import("nn.zig");
const g = @import("grad.zig");
const mul = g.mul;
const add = g.add;
const vari = g.variable_uuid;

const Layer = nn.Layer;

//nonlinear function to approximate
//x^2 + y^2 -3xy
pub fn to_approx(in: [2]f64) [1]f64 { //in grad val to make eval func easier
    return .{(in[0] * in[0]) + (in[1] * in[1]) + (-3 * in[0] * in[1])};
}

pub fn loss(expected: [1]g.GradVal, actual: [1]g.GradVal) g.GradVal {
    var dx = g.sub(expected[0], actual[0]);
    //just squared euclidean distance
    return mul(dx, dx);
}

pub fn randFromRange(range: [2]f64, random: rand.Random) f64 {
    return range[0] + (range[1] - range[0]) * random.float(f64);
}

pub fn main() !void {
    const nn_type = nn.NN(.{ Layer(2, 4, g.sigmoid), Layer(4, 1, g.sigmoid) }, 2, 1);

    var approx: nn_type = nn_type{};

    const steps = 100;
    const batch_size = 20;

    var rng = rand.DefaultPrng.init(0);
    const random = rng.random();
    const range: [2]f64 = .{ -100, 100 };
    for (0..steps) |step| {
        //randomly sample some points
        var batch_in: [batch_size][2]f64 = [_][2]f64{.{ 0, 0 }} ** batch_size;
        var batch_out: [batch_size][1]f64 = [_][1]f64{.{0}} ** batch_size;
        for (0..batch_size) |i| {
            batch_in[i] = .{ randFromRange(range, random), randFromRange(range, random) };
            batch_out[i] = to_approx(batch_in[i]);
        }

        var curr_loss = nn_type.train_step(&approx.layers, &batch_in, &batch_out, loss, 0.1);
        std.debug.print("Step {}, loss: {}\n", .{ step, curr_loss });
    }
}
