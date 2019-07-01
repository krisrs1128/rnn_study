
function rnn_factory(wx, wh, b) {
    function f(x, h) {
        // assuming one-dimensional x and h
        return Math.tanh(wx * x + wh * h + b);
    }
    return f;
}

// example for testing
var f_rnn = rnn_factory(-1, 1, 0);
