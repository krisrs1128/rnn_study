
function rnn_factory(wx, wh, b) {
    function f(x, h) {
        // assuming one-dimensional x and h
        return Math.tanh(wx * x + wh * h + b);
    }
    return f;
}

function warping_factory(a) {
    // for now, just linear warpings
    function c(times) {
        var result = [];
        for (var i = 0; i < times.length; i++) {
            result.push(a * times[i]);
        }
        return result;
    }
    return c;
}

function evaluate_dynamics(times, d0, x, f) {
    var result = [d0];
    for (var i = 1; i < N; i++) {
        var cur_x = x(times[i]),
            h = f_rnn(cur_x, result[i - 1]["h"]),
            h_next = f_rnn(cur_x, h); // at this x, h, where would dynamics want you to go?

        result.push({"time": times[i], "x": cur_x, "h": h, "h_next": h_next});
    }
    return result;
}

function draw_x_fun(elem, dynamics, scale) {
    var x_points = elem.selectAll("circles")
        .data(dynamics).enter()
        .append("circle")
        .attrs({
            "cx": (d) => scale.time(d.time),
            "cy": (d) => scale.value(d.x),
            "r": 1
        });
}

function draw_xh_scatter(elem, dynamics, scale) {
    var xh_segs = elem.selectAll("line")
        .data(dynamics).enter()
        .append("line")
        .attrs({
            "x1": (d) => scale.h(d["h"]),
            "x2": (d) => scale.h(d["h_next"]),
            "y1": (d) => scale.x(d["x"]),
            "y2": (d) => scale.x(d["x"]),
            "stroke": "#000",
            "stroke-width": 2,
            "stroke-opacity": 0.2
        });

    var xh_starts = elem.selectAll("circle")
        .data(dynamics).enter()
        .append("circle")
        .attrs({
            "cx": (d) => scale.h(d["h"]),
            "cy": (d) => scale.x(d["x"]),
            "r": 2
        });
}
