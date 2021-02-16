import numpy as np
import matplotlib.pyplot as plt
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from .interactive import view_stack
from .funcs import load_image
from .roi import def_roi
from .quantification import ImageQuant

"""
Todo: progress bar

"""


class ImageQuantGUI:
    """
    Graphical user interface for ImageQuant

    """

    def __init__(self):

        """
        Input data

        """

        self.file_path = None
        self.img = None
        self.cytbg = None
        self.membg = None

        """
        Parameters

        """
        self.stack = None
        self.ROI = None
        self.sigma = None
        self.nfits = None
        self.rol_ave = None
        self.start_frame = None
        self.end_frame = None
        self.interpolation_type = None
        self.iterations = None
        self.thickness = None
        self.freedom = None
        self.periodic = None
        self.parallel = None
        self.bg_subtract = None
        self.uni_cyt = None
        self.uni_mem = None
        self.mode = None

        """
        Help text

        """
        with open(os.path.dirname(os.path.realpath(__file__)) + '/gui_help.txt', 'r') as file:
            self.help_info = file.read()

        """
        Open window

        """
        self.window = tk.Tk()
        self.window.resizable(width=False, height=False)

        """
        Upper buttons

        """

        self.button_open = tk.Button(master=self.window, text='Import tif...', command=self.button_open_event)
        self.label_nframes = tk.Label(master=self.window, text='')

        self.button_cytbg = tk.Button(master=self.window, text='Import cytoplasmic profile...',
                                      command=self.button_cytbg_event)
        self.label_cytbg = tk.Label(master=self.window, text='')

        self.button_membg = tk.Button(master=self.window, text='Import membrane profile...',
                                      command=self.button_membg_event)
        self.label_membg = tk.Label(master=self.window, text='')

        """
        Basic parameters inputs

        """

        self.label_sigma = tk.Label(master=self.window, text='Sigma')
        self.entry_sigma = tk.Spinbox(master=self.window,
                                      values=["{:.1f}".format(i) for i in list(np.arange(0.1, 10, 0.1))])
        self.entry_sigma.delete(0, 'end')
        self.entry_sigma.insert(0, '3.0')

        self.label_nfits = tk.Label(master=self.window, text='Number of fits')
        self.entry_nfits = tk.Entry(master=self.window)
        self.entry_nfits.insert(0, '100')

        self.label_rolave = tk.Label(master=self.window, text='Roling average window')
        self.entry_rolave = tk.Spinbox(master=self.window, values=list(range(0, 110, 10)))
        self.entry_rolave.delete(0, 'end')
        self.entry_rolave.insert(0, '10')

        self.label_start = tk.Label(master=self.window, text='Start frame')
        self.entry_start = tk.Spinbox(master=self.window, values=list(range(0, 10, 1)))
        self.entry_start.delete(0, 'end')
        self.entry_start.insert(0, '0')

        self.label_end = tk.Label(master=self.window, text='End frame')
        self.entry_end = tk.Spinbox(master=self.window, values=list(range(0, 10, 1)))
        self.entry_end.delete(0, 'end')
        self.entry_end.insert(0, '0')

        self.label_iterations = tk.Label(master=self.window, text='Iterations')
        self.entry_iterations = tk.Spinbox(master=self.window, values=list(range(1, 100, 1)))
        self.entry_iterations.delete(0, 'end')
        self.entry_iterations.insert(0, '1')

        self.label_thickness = tk.Label(master=self.window, text='Thickness')
        self.entry_thickness = tk.Spinbox(master=self.window, values=list(range(10, 110, 10)))
        self.entry_thickness.delete(0, 'end')
        self.entry_thickness.insert(0, '50')

        self.label_freedom = tk.Label(master=self.window, text='ROI freedom')
        self.entry_freedom = tk.Spinbox(master=self.window,
                                        values=["{:.1f}".format(i) for i in list(np.arange(0, 51, 1))])
        self.entry_freedom.delete(0, 'end')
        self.entry_freedom.insert(0, '10')

        self.label_periodic = tk.Label(master=self.window, text='Periodic ROI')
        self.var_periodic = tk.IntVar(value=1)
        self.checkbutton_periodic = tk.Checkbutton(master=self.window, variable=self.var_periodic)

        self.label_bg = tk.Label(master=self.window, text='Subtract background')
        self.var_bg = tk.IntVar(value=1)
        self.checkbutton_bg = tk.Checkbutton(master=self.window, variable=self.var_bg)

        """
        Advanced parameters inputs

        """

        self.label_unicyt = tk.Label(master=self.window, text='Uniform cytoplasm')
        self.var_unicyt = tk.IntVar(value=0)
        self.checkbutton_unicyt = tk.Checkbutton(master=self.window, variable=self.var_unicyt)

        self.label_unimem = tk.Label(master=self.window, text='Uniform membrane')
        self.var_unimem = tk.IntVar(value=0)
        self.checkbutton_unimem = tk.Checkbutton(master=self.window, variable=self.var_unimem)

        """
        Lower buttons

        """

        self.button_view = tk.Button(master=self.window, text='View', command=self.button_view_event)
        self.button_ROI = tk.Button(master=self.window, text='Specify ROI', command=self.button_ROI_event)
        self.label_ROI = tk.Label(master=self.window, text='')
        self.button_run = tk.Button(master=self.window, text='Run quantification', command=self.button_run_event)
        self.label_running = tk.Label(master=self.window, text='')

        self.button_quant = tk.Button(master=self.window, text='View membrane quantification',
                                      command=self.button_quant_event)
        self.button_fits = tk.Button(master=self.window, text='View local fits', command=self.button_fits_event)
        self.button_seg = tk.Button(master=self.window, text='View segmentation', command=self.button_seg_event)
        self.button_save = tk.Button(master=self.window, text='Save to csv...', command=self.button_save_event)
        self.button_mode = tk.Button(master=self.window)
        self.button_help = tk.Button(master=self.window, text='Help', command=self.button_help_event)

        """
        Lay out grid

        """

        self.button_open.grid(row=0, column=0, sticky='W', padx=10, pady=5)
        self.label_nframes.grid(row=0, column=1, sticky='W', padx=10, pady=5)

        self.button_cytbg.grid(row=1, column=0, sticky='W', padx=10, pady=5)
        self.label_cytbg.grid(row=1, column=1, sticky='W', padx=10, pady=5)

        self.button_membg.grid(row=2, column=0, sticky='W', padx=10, pady=5)
        self.label_membg.grid(row=2, column=1, sticky='W', padx=10, pady=5)

        self.label_start.grid(row=3, column=0, sticky='W', padx=10)
        self.entry_start.grid(row=3, column=1, sticky='W', padx=10)
        self.label_end.grid(row=4, column=0, sticky='W', padx=10)
        self.entry_end.grid(row=4, column=1, sticky='W', padx=10)

        self.label_sigma.grid(row=5, column=0, sticky='W', padx=10)
        self.entry_sigma.grid(row=5, column=1, sticky='W', padx=10)
        self.label_nfits.grid(row=6, column=0, sticky='W', padx=10)
        self.entry_nfits.grid(row=6, column=1, sticky='W', padx=10)
        self.label_rolave.grid(row=7, column=0, sticky='W', padx=10)
        self.entry_rolave.grid(row=7, column=1, sticky='W', padx=10)

        self.label_iterations.grid(row=9, column=0, sticky='W', padx=10)
        self.entry_iterations.grid(row=9, column=1, sticky='W', padx=10)
        self.label_thickness.grid(row=10, column=0, sticky='W', padx=10)
        self.entry_thickness.grid(row=10, column=1, sticky='W', padx=10)
        self.label_freedom.grid(row=11, column=0, sticky='W', padx=10)
        self.entry_freedom.grid(row=11, column=1, sticky='W', padx=10)
        self.label_periodic.grid(row=12, column=0, sticky='W', padx=10, pady=3)
        self.checkbutton_periodic.grid(row=12, column=1, sticky='W', padx=10, pady=3)
        self.label_bg.grid(row=13, column=0, sticky='W', padx=10, pady=3)
        self.checkbutton_bg.grid(row=13, column=1, sticky='W', padx=10, pady=3)
        self.label_unicyt.grid(row=15, column=0, sticky='W', padx=10, pady=3)
        self.checkbutton_unicyt.grid(row=15, column=1, sticky='W', padx=10, pady=3)
        self.label_unimem.grid(row=16, column=0, sticky='W', padx=10, pady=3)
        self.checkbutton_unimem.grid(row=16, column=1, sticky='W', padx=10, pady=3)

        self.button_view.grid(row=19, column=0, sticky='W', padx=10, pady=5)
        self.button_ROI.grid(row=20, column=0, sticky='W', padx=10, pady=5)
        self.label_ROI.grid(row=20, column=1, sticky='W', padx=10, pady=5)
        self.button_run.grid(row=21, column=0, sticky='W', padx=10, pady=5)
        self.label_running.grid(row=21, column=1, sticky='W', padx=10, pady=5)
        self.button_quant.grid(row=22, column=0, sticky='W', padx=10, pady=5)
        self.button_fits.grid(row=23, column=0, sticky='W', padx=10, pady=5)
        self.button_seg.grid(row=24, column=0, sticky='W', padx=10, pady=5)
        self.button_mode.grid(row=24, column=1, sticky='E', padx=10, pady=5)
        self.button_save.grid(row=25, column=0, sticky='W', padx=10, pady=5)
        self.button_help.grid(row=25, column=1, sticky='E', padx=10, pady=5)

        """
        Start window

        """

        self.toggle_set1('disable')
        self.button_run.configure(state='disable')
        self.toggle_set2('disable')
        self.button_mode_to_basic()  # basic mode by default
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.window.mainloop()

    """
    Button functions

    """

    def button_open_event(self):
        root = tk.Tk()
        root.withdraw()
        root.update()
        file_path = filedialog.askopenfilename(master=root)
        self.file_path = file_path
        root.destroy()
        # self.window.lift()

        # Load image
        self.img = load_image(file_path)

        # Stack vs frame
        if len(self.img.shape) == 3:
            self.stack = True
        else:
            self.stack = False

        # Activate set 1
        self.toggle_set1('normal')

        # Disable run + set 2
        self.button_run.configure(state='disable')
        self.toggle_set2('disable')
        self.label_running.config(text='')
        self.label_ROI.config(text='')

        # Set frame ranges
        if self.stack:
            self.entry_start.configure(values=list(range(0, self.img.shape[0], 1)))
            self.entry_end.configure(values=list(range(0, self.img.shape[0], 1)))
            self.entry_start.delete(0, 'end')
            self.entry_start.insert(0, 0)
            self.entry_end.delete(0, 'end')
            self.entry_end.insert(0, str(self.img.shape[0] - 1))
            self.label_nframes.config(text='%s frames loaded' % self.img.shape[0])
        else:
            self.entry_start.delete(0, 'end')
            self.entry_start.insert(0, 0)
            self.entry_end.delete(0, 'end')
            self.entry_end.insert(0, 0)
            self.entry_start.configure(state='disable')
            self.entry_end.configure(state='disable')
            self.label_start.configure(state='disable')
            self.label_end.configure(state='disable')
            self.label_nframes.config(text='1 frame loaded')

    def button_cytbg_event(self):
        root = tk.Tk()
        root.withdraw()
        root.update()
        file_path = filedialog.askopenfilename(master=root)
        root.destroy()

        # Load cytbg
        self.cytbg = np.loadtxt(file_path)

        # Update window
        self.label_cytbg.config(text='Cytoplasmic profile loaded')
        if self.cytbg is not None and self.membg is not None:
            self.entry_sigma.configure(state='normal')
            self.label_sigma.configure(state='normal')
            self.entry_sigma.delete(0, 'end')
            self.entry_sigma.configure(state='disable')
            self.label_sigma.configure(state='disable')

    def button_membg_event(self):
        root = tk.Tk()
        root.withdraw()
        root.update()
        file_path = filedialog.askopenfilename(master=root)
        root.destroy()

        # Load cytbg
        self.membg = np.loadtxt(file_path)

        # Update window
        self.label_membg.config(text='Membrane profile loaded')
        if self.cytbg is not None and self.membg is not None:
            self.entry_sigma.configure(state='normal')
            self.label_sigma.configure(state='normal')
            self.entry_sigma.delete(0, 'end')
            self.entry_sigma.configure(state='disable')
            self.label_sigma.configure(state='disable')

    def button_help_event(self):
        popup = tk.Tk()
        popup.wm_title('Help')
        popup.geometry('500x750')
        popup.resizable(width=False, height=False)

        text_box = tk.Text(master=popup, wrap=tk.WORD)
        text_box.insert('1.0', self.help_info)
        text_box.config(state=tk.DISABLED)
        text_box.pack(expand=True, fill='both')
        popup.mainloop()

    def button_view_event(self):
        view_stack(self.img)

    def button_mode_to_basic(self):
        self.mode = 0
        self.window.title('Basic mode')

        self.button_mode.config(text='Advanced mode', command=self.button_mode_to_advanced)
        self.button_cytbg.grid_remove()
        self.label_cytbg.grid_remove()
        self.button_membg.grid_remove()
        self.label_membg.grid_remove()
        self.label_unicyt.grid_remove()
        self.checkbutton_unicyt.grid_remove()
        self.label_unimem.grid_remove()
        self.checkbutton_unimem.grid_remove()

        self.entry_sigma.configure(state='normal')
        self.label_sigma.configure(state='normal')
        self.entry_sigma.delete(0, 'end')
        self.entry_sigma.insert(0, '3.0')
        if self.img is None:
            self.entry_sigma.configure(state='disabled')
            self.label_sigma.configure(state='disabled')

    def button_mode_to_advanced(self):
        self.mode = 1
        self.window.title('Advanced mode')

        self.button_mode.config(text='Basic mode', command=self.button_mode_to_basic)
        self.button_cytbg.grid()
        self.label_cytbg.grid()
        self.button_membg.grid()
        self.label_membg.grid()
        self.label_unicyt.grid()
        self.checkbutton_unicyt.grid()
        self.label_unimem.grid()
        self.checkbutton_unimem.grid()

        if self.cytbg is not None and self.membg is not None:
            self.entry_sigma.configure(state='normal')
            self.label_sigma.configure(state='normal')
            self.entry_sigma.delete(0, 'end')
            self.entry_sigma.configure(state='disable')
            self.label_sigma.configure(state='disable')

    def button_ROI_event(self):
        self.ROI = def_roi(self.img, start_frame=int(self.entry_start.get()), end_frame=int(self.entry_end.get()),
                           periodic=bool(self.var_periodic.get()))
        if self.ROI is not None:
            self.label_ROI.configure(text='ROI saved')
            self.button_run.configure(state='normal')

    def button_run_event(self):

        # Get parameters
        if self.entry_sigma.get() != '':
            self.sigma = float(self.entry_sigma.get())
        self.nfits = int(self.entry_nfits.get())
        self.rol_ave = int(self.entry_rolave.get())
        self.start_frame = int(self.entry_start.get())
        self.end_frame = int(self.entry_end.get())
        self.iterations = int(self.entry_iterations.get())
        self.thickness = int(self.entry_thickness.get())
        self.freedom = float(self.entry_freedom.get())
        self.periodic = bool(self.var_periodic.get())
        self.bg_subtract = bool(self.var_bg.get())
        self.uni_cyt = bool(self.var_unicyt.get())
        self.uni_mem = bool(self.var_unimem.get())

        try:

            # Input
            if self.stack is True:
                inpt = self.img[self.start_frame:self.end_frame]
            else:
                inpt = self.img

            # Set up quantifier class
            if self.mode == 0:  # basic mode
                self.quantifier = ImageQuant(inpt, roi=self.ROI, thickness=self.thickness, rol_ave=self.rol_ave,
                                             nfits=self.nfits, sigma=self.sigma, iterations=self.iterations,
                                             periodic=self.periodic,
                                             bg_subtract=self.bg_subtract, uni_cyt=False, uni_mem=False,
                                             descent_steps=500, freedom=self.freedom, zerocap=False)

            else:  # advanced mode
                self.quantifier = ImageQuant(inpt, roi=self.ROI, thickness=self.thickness, rol_ave=self.rol_ave,
                                             nfits=self.nfits,
                                             iterations=self.iterations, periodic=self.periodic,
                                             bg_subtract=self.bg_subtract,
                                             cytbg=self.cytbg, membg=self.membg, uni_cyt=self.uni_cyt,
                                             uni_mem=self.uni_mem, sigma=self.sigma, descent_steps=500,
                                             freedom=self.freedom, zerocap=False)

            # Update window
            self.toggle_set2('disable')
            self.label_running.config(text='Running...')
            self.window.update()

            # Run
            self.quantifier.run()

            # Update window
            self.toggle_set2('normal')
            self.label_running.config(text='Complete!')

        except Exception as e:
            print(e)
            self.label_running.config(text='Failed (check terminal)')

    def button_quant_event(self):
        self.quantifier.plot_quantification()

    def button_fits_event(self):
        self.quantifier.plot_fits()

    def button_seg_event(self):
        self.quantifier.plot_segmentation()

    def button_save_event(self):
        # Pick save destination
        root = tk.Tk()
        root.withdraw()
        root.update()
        name = os.path.splitext(os.path.basename(os.path.normpath(self.file_path)))[0] + '.csv'
        f = filedialog.asksaveasfile(master=root, mode='w', initialfile=name)
        root.destroy()

        # Compile results
        res = self.quantifier.compile_res()

        # Save
        res.to_csv(f)

    """
    Enable / disable widgets

    """

    def toggle_set1(self, state):

        self.label_sigma.configure(state=state)
        self.entry_sigma.configure(state=state)
        self.label_nfits.configure(state=state)
        self.entry_nfits.configure(state=state)
        self.label_rolave.configure(state=state)
        self.entry_rolave.configure(state=state)
        self.label_start.configure(state=state)
        self.entry_start.configure(state=state)
        self.label_end.configure(state=state)
        self.entry_end.configure(state=state)

        self.label_iterations.configure(state=state)
        self.entry_iterations.configure(state=state)
        self.label_thickness.configure(state=state)
        self.entry_thickness.configure(state=state)
        self.label_freedom.configure(state=state)
        self.entry_freedom.configure(state=state)
        self.label_periodic.configure(state=state)
        self.checkbutton_periodic.configure(state=state)
        self.label_bg.configure(state=state)
        self.checkbutton_bg.configure(state=state)

        self.label_unicyt.configure(state=state)
        self.checkbutton_unicyt.configure(state=state)
        self.label_unimem.configure(state=state)
        self.checkbutton_unimem.configure(state=state)

        self.button_view.configure(state=state)
        self.button_ROI.configure(state=state)

    def toggle_set2(self, state):
        self.button_quant.configure(state=state)
        self.button_fits.configure(state=state)
        self.button_seg.configure(state=state)
        self.button_save.configure(state=state)

    """
    Shutdown

    """

    def on_closing(self):
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            # Close
            plt.close('all')
            self.window.destroy()
