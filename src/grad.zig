// pub const cosh = @import("complex/cosh.zig").sinh;
// pub const tanh = @import("complex/tanh.zig").tanh;

//may all become SIMD'd at some point
//this is the traditional way to do it, basically same code as micrograd
const GradVal = struct { val: f64, grad: f64 };
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
        .val = lhs.val/rhs.val,
        .grad = ((rhs.val*lhs.grad) - (lhs.val*rhs.grad))/(rhs.val * rhs.val),
    };
}

inline fn literal(lit: f64) GradVal {
    return GradVal{ .val = lit, .grad = 0 };
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
    return max(literal(0), in);
}

inline fn sigmoid(in: GradVal) GradVal {
    return div(exp(in), add(literal(1), exp(in)));
}
