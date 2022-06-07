"""Contain the classes to create cursors in the plotting"""
import matplotlib.pyplot as plt
__all__ = ['BlittedCursor']


class BlittedCursor:
    """
    A cross hair cursor using blitting for faster redraw.

    Adapted from matplolib documentation
    """

    def __init__(self, ax, color='g', verbose=True):
        # First see if we have an ax or a list
        try:
            len(ax)
        except TypeError:
            ax = [ax, ]
        self.ax = ax.flatten()
        self.n = len(self.ax)
        self.background = None
        self.horizontal_line = \
            [self.ax[j].axhline(color=color, lw=0.8, ls='--')
             for j in range(self.n)]
        self.vertical_line = \
            [self.ax[j].axvline(color=color, lw=0.8, ls='--')
             for j in range(self.n)]
        # text location in axes coordinates
        self.text = self.ax[0].text(0.72, 0.9, '',
                                    transform=self.ax[0].transAxes,
                                    color=color)
        self._creating_background = False
        self.ax[0].figure.canvas.mpl_connect('draw_event', self.on_draw)
        if verbose:
            print('You need to run:')
            print('blitted_cursor = BlittedCursor(ax)')
            print("fig.canvas.mpl_connect('motion_notify_event',")
            print("                       blitted_cursor.on_mouse_move)")

    def on_draw(self, event):
        self.create_new_background()

    def set_cross_hair_visible(self, visible):
        need_redraw = self.horizontal_line[0].get_visible() != visible
        for j in range(self.n):
            self.horizontal_line[j].set_visible(visible)
            self.vertical_line[j].set_visible(visible)
        self.text.set_visible(visible)
        return need_redraw

    def create_new_background(self):
        if self._creating_background:
            # discard calls triggered from within this function
            return
        self._creating_background = True
        self.set_cross_hair_visible(False)
        self.ax[0].figure.canvas.draw()
        self.background =\
            [self.ax[j].figure.canvas.copy_from_bbox(self.ax[j].bbox)
             for j in range(self.n)]
        self.set_cross_hair_visible(True)
        self._creating_background = False

    def on_mouse_move(self, event):
        if self.background is None:
            self.create_new_background()
        if not event.inaxes:
            need_redraw = self.set_cross_hair_visible(False)
            if need_redraw:
                for j in range(self.n):
                    self.ax[j].figure.canvas.restore_region(self.background[j])
                    self.ax[j].figure.canvas.blit(self.ax[j].bbox)
        else:
            self.set_cross_hair_visible(True)
            # update the line positions
            x, y = event.xdata, event.ydata

            self.text.set_text('x=%1.2f, y=%1.2f' % (x, y))
            for j in range(self.n):
                self.horizontal_line[j].set_ydata(y)
                self.vertical_line[j].set_xdata(x)
                self.ax[j].figure.canvas.restore_region(self.background[j])
                self.ax[j].draw_artist(self.horizontal_line[j])
                self.ax[j].draw_artist(self.vertical_line[j])
                self.ax[j].figure.canvas.blit(self.ax[j].bbox)

            self.ax[0].draw_artist(self.text)
