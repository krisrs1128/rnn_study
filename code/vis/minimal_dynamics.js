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
    .data(["x_fun", "x_star_fun", "hx_star_scatter", "hx_scatter"]).enter()
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
    c_warp = warping_factory(.5);

var dynamics = evaluate_dynamics(times, d0, x, f_rnn);
var warped_dynamics = evaluate_dynamics(c_warp(times), d0, x, f_rnn);
for (var i = 0; i < N; i++) {
    warped_dynamics[i]["time"] = times[i];
}

// Display these data
draw_x_fun(svg_elem.select("#x_fun"), dynamics, scales.x_fun);
draw_x_fun(svg_elem.select("#x_star_fun"), warped_dynamics, scales.x_fun);

draw_xh_scatter(svg_elem.select("#hx_scatter"), dynamics, scales.hx);
draw_xh_scatter(svg_elem.select("#hx_star_scatter"), warped_dynamics, scales.hx);
