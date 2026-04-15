import json
import os

import numpy as np

# --- Physical Constants (SI Units) ---
C0   = 299792458        # Speed of light (m/s)
EPS0 = 8.8541878128e-12 # Vacuum Permittivity (F/m)
MU0  = 1.25663706e-6    # Vacuum Permeability (H/m)

class Material:
    def __init__( self, name, epsilon=1.0, sigma=0.0, mu=1.0, loss=0.0, color=None, debye_poles=None ):
        """
        The Master Material Class for OpenEMS.
        :param name: String name (e.g., 'FR4')
        :param epsilon: Relative Permittivity (epsilon_r)
        :param sigma: DC Conductivity in S/m (kappa)
        :param mu: Relative Permeability (mu_r)
        :param loss: Loss Tangent (tan delta)
        :param color: List [R, G, B] (0-255)
        :param debye_poles: List of dicts [{'eps_s': val, 'tau': val}]
        """
        self.name = name
        self.epsilon = epsilon
        self.sigma = sigma
        self.mu = mu
        self.loss = loss
        self.color = color if color else [150, 150, 150]
        self.debye_poles = debye_poles if debye_poles else []

    # --- Library Management (JSON) ---

    @classmethod
    def from_library(cls, name, library_path="materials.json"):
        """Loads a material from a JSON file by its name."""

        if not os.path.exists(library_path):
            # Create a dummy file if it doesn't exist to avoid crashing
            # with open(library_path, 'w') as f:
                # json.dump({}, f)
            raise FileNotFoundError(f"Library {library_path} created empty. Add materials first.")

        with open(library_path, 'r') as f:
            db = json.load(f)

        if name not in db:
            raise KeyError(f"Material '{name}' not found in {library_path}")

        data = db[name]

        return cls(
            name=name,
            epsilon=data.get("epsilon") or data.get("epsilon_inf") or 1.0,
            sigma=data.get("sigma", 0.0),
            mu=data.get("mu", 1.0),
            loss=data.get("loss_tangent", data.get("loss", 0.0)),
            color=data.get("color"),
            debye_poles=data.get("debye_poles", [])
        )

    def save_to_library(self, library_path="materials.json"):
        """Saves current material instance into the JSON library."""

        db = {}
        if os.path.exists(library_path):
            with open(library_path, 'r') as f:
                try:
                    db = json.load(f)
                except json.JSONDecodeError:
                    db = {}

        db[self.name] = {
            "epsilon": self.epsilon,
            "sigma": self.sigma,
            "mu": self.mu,
            "loss": self.loss,
            "color": self.color,
            "debye_poles": self.debye_poles
        }

        with open(library_path, 'w') as f:
            json.dump(db, f, indent=4)

    # --- Physics & Solver Calculations ---

    def get_kappa_at(self, f):
        """
        Calculates total effective conductivity (kappa) in S/m.
        Combines DC sigma and frequency-dependent dielectric loss.
        """
        omega = 2 * np.pi * f
        # sigma_loss = omega * epsilon_0 * epsilon_r * tan_delta
        sigma_ac = omega * EPS0 * self.epsilon * self.loss
        return self.sigma + sigma_ac

    def get_complex_epsilon(self, f):
        """Calculates complex permittivity: eps_real - j*eps_imag"""
        omega = 2 * np.pi * f
        eps_complex = self.epsilon + 0j

        # Add Debye contributions (Dispersion)
        for pole in self.debye_poles:
            # delta_eps = eps_static - eps_infinity
            # We assume self.epsilon is our epsilon_infinity
            deps = pole['eps_s'] - self.epsilon
            eps_complex += deps / (1 + 1j * omega * pole['tau'])

        # Add total conductivity contribution
        kappa_total = self.get_kappa_at(f)
        eps_complex -= 1j * kappa_total / (omega * EPS0)

        return eps_complex

    def get_lambda_at(self, f, unit=1e-3):
        """
        Calculates wavelength inside the material.
        :param f: Frequency in Hz
        :param unit: Conversion factor (1e-3 for mm, 1 for meters)
        """
        if f <= 0:
            return np.inf

        # Phase velocity v = c / sqrt(er * ur)
        v_phase = C0 / np.sqrt(self.epsilon * self.mu)
        return (v_phase / f) / unit


    def get_cells_pe_lambda( self, f, cells_per_wavelength=15 ):
        """
        get optimal mesh cells count per wavelength
        """
        return self.get_lambda_at(self, f) / cells_per_wavelength


    def get_skin_depth(self, f):
        """Calculates skin depth (delta) in meters."""
        kappa = self.get_kappa_at(f)
        if kappa <= 0:
            return np.inf
        return np.sqrt(1 / (np.pi * f * (self.mu * MU0) * kappa))

    # --- OpenEMS Integration Helper ---
    def add_to_csx(self, CSX, f0=None):
        """
        Registers material. f0 is used to calculate effective sigma
        from the loss tangent for non-dispersive simulations.
        """
        # If f0 is provided, calculate the effective conductivity (sigma + AC loss)
        # If not, fall back to the DC sigma

        if self.sigma > 1e6:
            # AddMetal does not require epsilon or sigma to be set
            mat = CSX.AddMetal(self.name)
            return mat

        target_sigma = self.get_kappa_at(f0) if f0 else self.sigma

        if not self.debye_poles:
            mat = CSX.AddMaterial(
                self.name,
                epsilon=self.epsilon,
                sigma=target_sigma, # Use the calculated kappa
                color=self.color
            )
        else:
            # Debye materials calculate frequency response internally in FDTD
            mat = CSX.AddDebyeMaterial(
                self.name,
                epsilon_inf=self.epsilon,
                sigma=self.sigma,
                color=self.color
            )
            for pole in self.debye_poles:
                deps = pole['eps_s'] - self.epsilon
                mat.AddPole(deps, pole['tau'])

        if self.mu != 1.0:
            mat.SetProperty(mu=self.mu)

        return mat

    def __repr__(self):
        return f"Material(Name='{self.name}', Eps={self.epsilon}, Kappa={self.sigma})"
