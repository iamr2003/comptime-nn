// pub const cosh = @import("complex/cosh.zig").sinh;
// pub const tanh = @import("complex/tanh.zig").tanh;
const std = @import("std");

//may all become SIMD'd at some point
//this is the traditional way to do it, basically same code as micrograd
const GradVal = struct {
    val: f64,
    grad: f64,
    // with_respect_to: ?*const []u8 = null,
};

//need to store NAME of var,
//still a lot of questions about multi dimensions with weights

//give basic binary ops, building up in the traditional way
inline fn add(lhs: GradVal, rhs: GradVal) GradVal {
    return GradVal{ .val = lhs.val + rhs.val, .grad = lhs.grad + rhs.grad };
}

inline fn sub(lhs: GradVal, rhs: GradVal) GradVal {
    return GradVal{ .val = lhs.val - rhs.val, .grad = lhs.grad - rhs.grad };
}

inline fn mul(lhs: GradVal, rhs: GradVal) GradVal {
    return GradVal{
        .val = lhs.val * rhs.val,
        .grad = lhs.grad * rhs.val + lhs.val * rhs.grad,
    };
}

inline fn div(lhs: GradVal, rhs: GradVal) GradVal {
    return GradVal{
        .val = lhs.val / rhs.val,
        .grad = ((rhs.val * lhs.grad) - (lhs.val * rhs.grad)) / (rhs.val * rhs.val),
    };
}

//goal is a lot of inefficiency compiles out
//we are going to duplicate expression a million times, perhaps leaving tree intact would be more helpful
inline fn variable(val: f64, name: []const u8, respect: []const u8) GradVal {
    return GradVal{ .val = val, .grad = if (std.mem.eql(u8,name, respect)) 1 else 0 };
}

inline fn literal(val: f64) GradVal {
    return GradVal{ .val = val, .grad = 0 };
}

inline fn max(lhs: GradVal, rhs: GradVal) GradVal {
    if (lhs.val < rhs.val) {
        return GradVal{ .val = rhs.val, .grad = rhs.grad };
    } else {
        return GradVal{ .val = lhs.val, .grad = lhs.grad };
    }
}

inline fn min(lhs: GradVal, rhs: GradVal) GradVal {
    if (lhs.val > rhs.val) {
        return GradVal{ .val = rhs.val, .grad = rhs.grad };
    } else {
        return GradVal{ .val = lhs.val, .grad = lhs.grad };
    }
}

//can write out arbitrary ones, but this is easier
inline fn exp(in: GradVal) GradVal {
    return GradVal{ .val = @exp(in.val), .grad = @exp(in.val) };
}

// inline fn tanhg(in:GradVal)GradVal{
//     return GradVal{
//         .val = tanh(in.val),
//         .grad = sech(in.val)
//     };
// }

//need exp for sigmoid
//tanh, ELU other common ones

inline fn relu(in: GradVal) GradVal {
    return max(variable(0), in);
}

inline fn sigmoid(in: GradVal) GradVal {
    return div(exp(in), add(variable(1), exp(in)));
}

//examples
fn square(val: f64) GradVal {
    return mul(variable(val, "x", "x"), variable(val, "x", "x"));
}

fn abs(val: f64) GradVal {
    return max(variable(val, "x", "x"), mul(literal(-1), variable(val, "x", "x")));
}

test "simple single var" {
    try std.testing.expectEqual(square(2).grad, (GradVal{ .val = 4, .grad = 4 }).grad);
    try std.testing.expectEqual(square(5), GradVal{ .val = 25, .grad = 10 });

    try std.testing.expectEqual(abs(7), GradVal{ .val = 7, .grad = 1 });
    try std.testing.expectEqual(abs(-7), GradVal{ .val = 7, .grad = -1 });
}
