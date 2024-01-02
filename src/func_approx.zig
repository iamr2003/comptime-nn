const std = @import("std");
const rand = std.rand;
const nn = @import("nn.zig");
const g = @import("grad.zig");
const mul = g.mul;
const add = g.add;

const Layer = nn.Layer;

//nonlinear function to approximate
//x^2 + y^2 -3xy
pub fn to_approx(in: [2]f64) [1]f64 { //in grad val to make eval func easier
    // return .{(in[0] * in[0]) + (in[1] * in[1]) + (-3 * in[0] * in[1])};
    //simple linear function
    return .{in[0] + in[1]};
}

pub fn loss(expected: [1]g.GradVal, actual: [1]g.GradVal) g.GradVal {
    std.debug.assert(expected[0].val != std.math.nan_f64);
    std.debug.assert(expected[0].val != std.math.nan_f64);

    var dx = g.sub(expected[0], actual[0]);
    //just squared euclidean distance
    var out = mul(dx, dx);
    std.debug.assert(out.val >= 0);
    std.debug.assert(out.grad != std.math.nan_f64);
    return out;
}

pub fn randFromRange(range: [2]f64, random: rand.Random) f64 {
    return range[0] + (range[1] - range[0]) * random.float(f64);
}

//this task might be way too hard for a bad slow model like this
//might switch to something simple like binary ops eventually
//am also curious how much faster simd is
// my loss function might also suck
pub fn main() !void {
    const nn_type = nn.NN(
        .{
            Layer(2, 16, g.relu),
            Layer(16, 16, g.relu),
            Layer(16, 1, g.relu),
        },
        2,
        1,
    );

    var approx: nn_type = nn_type{};

    const steps = 10000;
    const batch_size = 100;

    var rng = rand.DefaultPrng.init(0);
    const random = rng.random();
    //see if can approx over small domain
    //possible it should also be in the
    const range: [2]f64 = .{ -10, 10 };
    for (0..steps) |step| {
        //randomly sample some points
        var batch_in: [batch_size][2]f64 = [_][2]f64{.{ 0, 0 }} ** batch_size;
        var batch_out: [batch_size][1]f64 = [_][1]f64{.{0}} ** batch_size;
        for (0..batch_size) |i| {
            batch_in[i] = .{ randFromRange(range, random), randFromRange(range, random) };
            batch_out[i] = to_approx(batch_in[i]);
        }

        //rerun on same data for awhile
        const batch_cycles = 1;
        for (0..batch_cycles) |batch_cycle| {
            //could use an adaptive learning rate
            var curr_loss = nn_type.train_step(
                &approx.layers,
                &batch_in,
                &batch_out,
                loss,
                // (1.0 / @as(f64 , @floatFromInt(step)))
                // 0.00001
                1 - (0.9 * @as(f64, @floatFromInt(step)) / 100),
            );
            std.debug.print("Step {}.{}, loss: {}\n", .{ step, batch_cycle, curr_loss });
        }
    }
}
