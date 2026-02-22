import matplotlib as mpl
from matplotlib import gridspec
import matplotlib.pyplot as plt
from cycler import cycler
import sys
import numpy as np
import pandas as pd
import seaborn as sns

class BasePlotStyler:
    """
    A base class for applying publication-quality Matplotlib styles.
    
    This class handles common utilities like size conversion and
    color palettes. It is intended to be inherited by journal-specific
    styler classes (e.g., `PlosStyler`, `NatureStyler`).
    """
    
    # --- Class-level constants ---
    MM_TO_INCH = 1 / 25.4  # Conversion factor from mm to inches
    
    # Colorblind-friendly Wong palette
    # See Wong, B. (2011). Points of view: Color blindness. 
    # URL: https://www.nature.com/articles/nmeth.1618
    WONG_COLORS = [
        "#e69f00",  # Orange
        "#56b4e9",  # Sky Blue
        "#009e73",  # Bluish Green
        "#0072b2",  # Blue
        "#d55e00",  # Vermillion
        "#cc79a7",  # Reddish Purple
        "#f0e442"   # Yellow
    ]
    
    # --- NEW: Paul Tol's Color Schemes ---
    PAUL_TOL_BRIGHT_COLORS = [
        '#4477AA',  # Blue
        '#EE6677',  # Red
        '#228833',  # Green
        '#CCBB44',  # Yellow
        '#66CCEE',  # Cyan
        '#AA3377',  # Purple
        '#BBBBBB'   # Grey
    ]
    
    PAUL_TOL_MUTED_COLORS = [
        '#88CCEE',  # Cyan
        '#CC6677',  # Rose
        '#DDCC77',  # Sand
        '#117733',  # Green
        '#332288',  # Indigo
        '#AA4499',  # Purple
        '#44AA99',  # Teal
        '#999933',  # Olive
        '#882255',  # Wine
        '#BBBBBB'   # Grey
    ]
    
    # A simple, clear set of linestyles
    LINE_STYLES_LONG = [
         '-',           # Solid
         '--',          # Dashed
         '-.',          # Dash-dot
         ':',           # Dotted
         (0, (5, 5)),   # Loosely dashed
         (0, (3, 1, 1, 1)), # Dash-dot-dot
         (0, (5, 1)),   # Long dash, short gap
         (0, (1, 1)),   # Densely dotted
         (0, (3, 5, 1, 5)), # Complex 1
         (0, (5, 10))   # Long dash, long gap
    ]
    
    def __init__(self, font_size=9, palette='wong'):
        """
        Initializes the styler with a base font size.

        Args:
            font_size (int, optional): The base font size to use. Defaults to 9.
            palette (str, optional): The color palette to use.
                One of ['wong', 'tol-bright', 'tol-muted'].
                Defaults to 'wong'.
        """
        if not 6 <= font_size <= 14:
            print(f"Warning: font_size={font_size} is outside a typical range.",
                  file=sys.stderr)
        self.font_size = font_size
        self.label_size = font_size + 2
        
        # Select color palette
        if palette == 'wong':
            self.palette_colors = self.WONG_COLORS
        elif palette == 'tol-bright':
            self.palette_colors = self.PAUL_TOL_BRIGHT_COLORS
        elif palette == 'tol-muted':
            self.palette_colors = self.PAUL_TOL_MUTED_COLORS
        else:
            raise ValueError(
                f"Palette '{palette}' not recognized. "
                "Choose from ['wong', 'tol-bright', 'tol-muted']."
            )
        
        # Create the instance-level cycler
        self.plot_cycler = cycler(color=self.palette_colors)
        
        self.single_col_width_mm = 89  
        self.max_width_mm = 183

    def get_figsize_mm(self, width_mm, height_mm):
        """
        Converts mm dimensions to a (width, height) tuple in inches,
        which is required by Matplotlib.

        Args:
            width_mm (float): Desired figure width in millimeters.
            height_mm (float): Desired figure height in millimeters.

        Returns:
            tuple: (width_in_inches, height_in_inches)
        """
        return (width_mm * self.MM_TO_INCH, height_mm * self.MM_TO_INCH)

    def create_figure(self, width_mm, height_mm, rows=1, cols=1, **kwargs):
        """
        A helper function to generate a Matplotlib figure and axes
        with a specific size in millimeters.
        
        All additional keyword arguments (`sharex`, `sharey`, `gridspec_kw`,
        `constrained_layout`, etc.) are passed directly to `plt.subplots`.

        Args:
            width_mm (float): Desired figure width in millimeters.
            height_mm (float): Desired figure height in millimeters.
            rows (int, optional): Number of subplot rows. Defaults to 1.
            cols (int, optional): Number of subplot columns. Defaults to 1.
            **kwargs: Additional arguments for `plt.subplots()`.

        Returns:
            tuple: (matplotlib.figure.Figure, matplotlib.axes.Axes)
                   A tuple containing the figure and axes objects.
        """
        figsize_inches = self.get_figsize_mm(width_mm, height_mm)
        
        # Pass all other kwargs directly to subplots
        fig, axs = plt.subplots(
            rows, 
            cols, 
            figsize=figsize_inches, 
            **kwargs
        )
        return fig, axs

    def apply_style(self):
        """
        Applies the journal-specific style.
        This method MUST be overridden by a child class.
        """
        raise NotImplementedError("Child classes must implement this method.")

    def reset_style(self):
        """Resets all Matplotlib styles to their default values."""
        mpl.rcParams.update(mpl.rcParamsDefault)
        print("Matplotlib style reset to default.")


class PlosStyler(BasePlotStyler):
    """
    Applies Matplotlib styles compliant with PLOS journals.
    """
    PLOS_COL_WIDTH_MM = 132
    PLOS_MAX_WIDTH_MM = 190.5
    ALLOWED_FONTS = ['Arial', 'Times New Roman', 'Times', 'Symbol']

    def __init__(self, font_size=9, palette='wong'):
        super().__init__(font_size, palette) # Call parent __init__
        self.single_col_width_mm = self.PLOS_COL_WIDTH_MM
        self.max_width_mm = self.PLOS_MAX_WIDTH_MM

    def apply_style(self, font_family='Arial'):
        """
        Applies the global Matplotlib style for PLOS.

        Args:
            font_family (str, optional): The font to use. Must be
                                         'Arial' or 'Times New Roman'.
        """
        if font_family not in self.ALLOWED_FONTS:
            raise ValueError(
                f"Font '{font_family}' not in PLOS allowed list: "
                f"{self.ALLOWED_FONTS}"
            )

        style_dict = {
            "font.size": self.font_size,
            "axes.titlesize": self.font_size,
            "axes.labelsize": self.font_size,
            "ytick.labelsize": self.font_size,
            "xtick.labelsize": self.font_size,
            "legend.fontsize": self.font_size,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "figure.figsize": self.get_figsize_mm(self.PLOS_COL_WIDTH_MM, 150),
            "savefig.dpi": 300,
            "savefig.format": "tif",
            "savefig.bbox": "tight",
            "figure.dpi": 300,
            "lines.linewidth": 1.5,
            "axes.grid": False,
            "axes.prop_cycle": self.plot_cycler,  # Use instance cycler
            "text.usetex": False,
            "mathtext.fontset": "custom",
        }

        if font_family == 'Arial':
            style_dict.update({
                # 'font.family': 'sans-serif',
                # 'font.sans-serif': ['Arial'],
                'mathtext.rm': 'Arial',
                'mathtext.it': 'Arial:italic',
                'mathtext.bf': 'Arial:bold',
            })
        elif font_family in ['Times New Roman', 'Times']:
            style_dict.update({
                'font.family': 'serif',
                'font.serif': ['Times New Roman', 'Times'],
                'mathtext.rm': 'Times New Roman',
                'mathtext.it': 'Times New Roman:italic',
                'mathtext.bf': 'Times New Roman:bold',
            })

        mpl.rcParams.update(style_dict)
        print(f"Applied PLOS style: {font_family}, {self.font_size}pt.")


class NatureStyler(BasePlotStyler):
    """
    Applies Matplotlib styles compliant with Nature journals.
    (Note: This is an example; always check current guidelines)
    """
    NATURE_COL_WIDTH_MM = 89
    NATURE_MAX_WIDTH_MM = 183
    ALLOWED_FONTS = ['Arial', 'Helvetica']

    def __init__(self, font_size=9, palette='wong'):
        super().__init__(font_size, palette) # Call parent __init__
        self.single_col_width_mm = self.NATURE_COL_WIDTH_MM
        self.max_width_mm = self.NATURE_MAX_WIDTH_MM

    def apply_style(self, font_family='Arial'):
        """
        Applies a global Matplotlib style for Nature.

        Args:
            font_family (str, optional): The font to use. Defaults to 'Arial'.
        """
        if font_family not in self.ALLOWED_FONTS:
            raise ValueError(
                f"Font '{font_family}' not in Nature allowed list: "
                f"{self.ALLOWED_FONTS}"
            )
            
        style_dict = {
            'font.family': 'sans-serif',
            'font.sans-serif': [font_family, 'sans-serif'],
            'font.size': self.font_size,
            'axes.titlesize': self.font_size,
            'axes.labelsize': self.font_size,
            'ytick.labelsize': self.font_size,
            'xtick.labelsize': self.font_size,
            'legend.fontsize': self.font_size,
            
            'figure.figsize': self.get_figsize_mm(self.NATURE_COL_WIDTH_MM, 100),
            
            'savefig.dpi': 300,
            'savefig.format': 'pdf', # Nature often prefers vector formats
            'savefig.bbox': 'tight',
            
            'figure.dpi': 150,
            'lines.linewidth': 1.0, # Nature plots are often finer
            'axes.grid': False,
            'axes.prop_cycle': self.plot_cycler, # Use instance cycler
            'text.usetex': False,
            'mathtext.fontset': 'custom',
            'mathtext.rm': font_family,
            'mathtext.it': f'{font_family}:italic',
            'mathtext.bf': f'{font_family}:bold',
        }
        
        mpl.rcParams.update(style_dict)
        print(f"Applied Nature style: {font_family}, {self.font_size}pt.")

# --- Example Usage ---
if __name__ == "__main__":

    # Check if the attributes exist
    print(f"Does 'rcParamsDefault' exist?     {hasattr(mpl, 'rcParamsDefault')}")
    print(f"Does 'rc_params_defaults' exist? {hasattr(mpl, 'rc_params_defaults')}")

    # Check if they are the exact same object
    if hasattr(mpl, 'rcParamsDefault') and hasattr(mpl, 'rc_params_defaults'):
        print(f"Are they the same object?        {mpl.rcParamsDefault is mpl.rc_params_defaults}")
    # --- Example 1: Your PLOS Figure ---
    print("--- Creating PLOS Figure ---")
    
    # 1. Initialize the specific styler
    # *** NEW: Select the 'tol-bright' palette ***
    plos_styler = PlosStyler(font_size=9, palette='tol-bright')
    
    # 2. Apply the global style for this script
    plos_styler.apply_style(font_family='Arial')

    # 3. Use the new figure generator
    # This now perfectly handles your exact, complex request
    fig, axs = plos_styler.create_figure(
        width_mm=90, 
        height_mm=170, 
        rows=5, 
        cols=2, 
        sharex=True, 
        sharey=True, 
        gridspec_kw={'width_ratios': [1, 1.2]}, 
        constrained_layout=True
    )
    
    fig.suptitle("PLOS-Styled Figure (90mm x 170mm)")
    
    # 'axs' is a 5x2 numpy array, plot as usual


    axs[0, 1].legend() # Add one legend
    
    # save_filename_plos = "Example_PLOS_Figure.tif"
    # fig.savefig(save_filename_plos)
    # print(f"Saved PLOS figure to '{save_filename_plos}'")


    # --- Example 2: Nature Figure (to show the difference) ---
    print("\n--- Creating Nature Figure ---")
    
    # 1. Initialize a different styler
    # *** NEW: Select the 'tol-muted' palette and smaller font ***
    nature_styler = NatureStyler(font_size=8, palette='tol-muted')
    
    # 2. Apply its style
    nature_styler.apply_style(font_family='Arial')
    
    # 3. Create a simple, single-column figure
    fig_nat, ax_nat = nature_styler.create_figure(
        width_mm=nature_styler.NATURE_COL_WIDTH_MM, # Use class constant
        height_mm=60,
        constrained_layout=True
    )
    
    ax_nat.set_title("Nature-Styled Figure (89mm wide)")
    ax_nat.plot(x, np.cos(x), label="Cosine")
    ax_nat.set_xlabel("X-Axis")
    ax_nat.set_ylabel("Y-Axis")
    ax_nat.legend()
    
    save_filename_nat = "Example_Nature_Figure.pdf"
    fig_nat.savefig(save_filename_nat)
    print(f"Saved Nature figure to '{save_filename_nat}'")
    

   
    # 4. Reset style when done
    print("\nResetting style to default...")
    plos_styler.reset_style()
