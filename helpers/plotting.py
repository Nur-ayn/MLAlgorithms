import numpy as np

def adjust_yerr(err, y_error):
    ln, (err_top, err_bot), (bars,) = err
    y_base = ln.get_ydata()
    x_base = ln.get_xdata()

    yerr_top = y_base + y_error
    yerr_bot = y_base - y_error

    err_top.set_ydata(yerr_top)
    err_bot.set_ydata(yerr_bot)

    new_segments = [np.array([[x, yt], [x, yb]]) for
                    x, yt, yb in zip(x_base, yerr_top, yerr_bot)]
   
    bars.set_segments(new_segments)