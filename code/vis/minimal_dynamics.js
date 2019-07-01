/* Drawing 1D Dynamics
 *
 * Mostly a warmup for the more complex visualizations. At the moment, just
 * trying to get a static view of the dynamics in a 1D RNN with fixed dynamics.
 */

// Define background
var height = 300,
    width = 500;

var svg_elem = d3.select("body")
    .append("svg")
    .attrs({"width": width,
           "height": height});

svg_elem.selectAll("g")
    .data(["x_fun", "x_star_fun", "hx_scatter"]).enter()
    .append("g")
    .attr("id", (d) => d);

var scales = {
    "x_fun": {
        "time": d3.scaleLinear()
            .range([400, 500]),
        "value": d3.scaleLinear()
            .domain([-1, 1])
            .range([100, 0])
    },
    "hx": {
        "h": d3.scaleLinear()
            .domain([-1, 1])
            .range([0, 300]),
        "x": d3.scaleLinear()
            .domain([-1, 1])
            .range([200, 0])
    }
};

// Simulate some data, just for testing
var times = d3.range(0, 1, 0.01);
var N = times.length;

var d0 = {"time": 0, "x": 0, "h": 0, "h_next": 0},
    x = (t) => Math.sin(2 * Math.PI * t),
    f_rnn = rnn_factory(-1, 1, 0),
    c_warp = warping_factory(2);

var dynamics = evaluate_dynamics(times, d0, x, f_rnn);
var warped_dynamics = evaluate_dynamics(c_warp(times), d0, x, f_rnn);
for (var i = 0; i < N; i++) {
    warped_dynamics[i]["time"] = times[i];
}

// Display these data
draw_x_fun(svg_elem.select("#x_fun"), dynamics, scales.x_fun);
draw_x_fun(svg_elem.select("#x_star_fun"), warped_dynamics, scales.x_fun);

var xh_segs = svg_elem.select("#hx_scatter")
    .selectAll("line")
    .data(dynamics).enter()
    .append("line")
    .attrs({
        "x1": (d) => scales.hx.h(d["h"]),
        "x2": (d) => scales.hx.h(d["h_next"]),
        "y1": (d) => scales.hx.x(d["x"]),
        "y2": (d) => scales.hx.x(d["x"]),
        "stroke": "#000",
        "stroke-width": 2,
        "stroke-opacity": 0.2
    });

var xh_starts = svg_elem.select("#hx_scatter")
    .selectAll("circle")
    .data(dynamics).enter()
    .append("circle")
    .attrs({
        "cx": (d) => scales.hx.h(d["h"]),
        "cy": (d) => scales.hx.x(d["x"]),
        "r": 2
    });
