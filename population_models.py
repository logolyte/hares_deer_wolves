#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
population_models.py

Simulates ecosystems with hares, deer and wolves. Reproduction, predation, aging and death are included
in the simulations.

Date: 2025-05-01
Author: Love Sundin
"""

import numpy
import scipy
from matplotlib import pyplot
from matplotlib import animation


def plot_concentrations(time_array, concentration_matrix, species_names=None,
                        title=None, filename=None, time_unit=None,
                        concentration_unit=None, figsize=(6, 5), colors=None,
                        legend_columns=None):
    """
    Function that plots concentrations for a set of species over time.

    Parameters
    ----------
    time_array : numpy.ndarray
        1-dimensional array with time points at which concentrations should
        be plotted.
    concentration_matrix : numpy.ndarray
        Array with shape (species, timepoints) or (timepoints, species)
        containing concentrations.
    species_names : array_like, optional
        List with names of all plotted species to be shown in the legend.
        Default is None, meaning no legend will be plotted.
    title : string, optional
        Title of the plot. Default is None, meaning no title will be displayed.
    filename : string, optional
        File name including path to which the plot will be saved. Default is
        None, meaning the plot will not be saved.
    time_unit : string, optional
        Unit for time. Default is None.
    concentration_unit : string
        Unit for concentrations. Default is None.
    figsize : tuple, optional
        Tuple of length 2 with width and height of the figure in inches.
        Default is (6, 5).
    colors : list, optional
        List with one color per species plotted determining the color of the
        lines. Colors should be interpretable by Pyplot. Default is None,
        meaning the default Pyplot colors will be used.
    legend_columns : int, optional
        Number of columns to show in the legend. Default is None, meaning
        each species will have its own column.

    Returns
    -------
    figure : pyplot.Figure
        Figure with the plotted concentrations.
    """

    # Make sure the concentration matrix has shape (species, timepoints)
    if numpy.shape(concentration_matrix)[1] != numpy.shape(time_array)[0] and\
            numpy.shape(concentration_matrix)[0] == numpy.shape(time_array)[0]:
        concentration_matrix = concentration_matrix.T

    # Set default values for optional parameters
    if legend_columns == None:
        legend_columns = numpy.shape(concentration_matrix)[1]
    if species_names == None:
        make_legend = False
    else:
        make_legend = True
    species = len(concentration_matrix)
    if species_names == None:
        species_names = [None for species_index in range(species)]
    if colors == None:
        colors = [None for species_index in range(species)]

    # Plot concentrations
    figure, axis = pyplot.subplots(figsize=figsize)
    for species_concentration, species_name, species_color in zip(concentration_matrix, species_names, colors):
        axis.plot(time_array, species_concentration,
                  label=species_name, color=species_color)

    # Format plot
    fontsize = 15
    x_label = "Time"
    if time_unit != None:
        x_label += f" ({time_unit})"
    axis.set_xlabel(x_label, fontsize=fontsize)
    y_label = "Concentration"
    if concentration_unit != None:
        y_label += f" ({concentration_unit})"
    axis.set_ylabel(y_label, fontsize=fontsize)
    axis.grid()
    if make_legend:
        axis.legend(fontsize=fontsize, frameon=False, loc="upper left", bbox_to_anchor=(0, 1.13, 1, 0), ncol=legend_columns, mode="expand",
                    borderaxespad=0, handlelength=1)
    axis.set_title(title, fontsize=fontsize, y=1.12)
    axis.tick_params(labelsize=13)
    figure.tight_layout()

    # Save figure if a filename is set
    if filename != None:
        figure.savefig(filename)

    return figure


def integrate_system(derivatives_function, initial_concentrations,
                     parameters=None, time_grid=numpy.linspace(0, 10, 100)):
    """
    Integrates a system of ordinary differential equations describing
    concentrations of species over time. Uses scipy.solve_ivp with the default
    method RK45.

    Parameters
    ----------
    derivatives_function : function
        Function describing concentration derivatives as a function of
        time and concentrations. Its first argument should be time as a float
        and the second argument should be a 1D array_like containing all
        concentrations. It can have additional parameters. Should return an
        array_like with concentration derivatives for each function.
    initial_concentrations : array_like
        Initial concentrations for all species.
    parameters : list, optional
        List of additional parameters that are passed to derivatives_function.
        The default is None.
    time_grid : numpy.ndarray, optional
        Array of time points for which concentrations are returned.
        The default is numpy.linspace(0, 10, 100).

    Returns
    -------
    time_array : numpy.ndarray
        Array of all time points for which concentrations are calculated.
        Equal to time_grid.
    concentration_matrix : numpy.ndarray
        Matrix of shape (species, time_points) containing concentrations for all species at
        all time points in time_array.

    """

    time_span = [time_grid[0], time_grid[-1]]
    solution = scipy.integrate.solve_ivp(derivatives_function, time_span,
                                         initial_concentrations,
                                         args=parameters,
                                         t_eval=time_grid)
    return solution.t, solution.y


species_names = ["Grass", "Hares", "Deer", "Wolves"]
species = len(species_names)
species_colors = ["tab:green", "tab:orange", "tab:red", "tab:blue"]

# Concentration derivatives


def species_derivatives(time, concentrations):
    dG = alpha_G*concentrations[0]*(1-concentrations[0]) - e_H*concentrations[1]*(concentrations[0]/(
        concentrations[0]+K_GH)) - e_D*concentrations[2]*(concentrations[0]/(concentrations[0]+K_GD))
    dH = alpha_H*concentrations[1]*concentrations[0]/(concentrations[0]+K_GH) - p_H*concentrations[3]*(
        concentrations[1]/(concentrations[1]+K_HW)) - delta_H*concentrations[1] - c_H*concentrations[1]**2
    dD = alpha_D*concentrations[2]*concentrations[0]/(concentrations[0]+K_GD) - p_D*concentrations[3]*(
        concentrations[2]/(concentrations[2]+K_DW)) - delta_D*concentrations[2] - c_D*concentrations[2]**2
    dW = alpha_W*concentrations[3]*(p_H*concentrations[1]/(concentrations[1]+K_HW)+p_D*concentrations[2]/(
        concentrations[2]+K_DW)) - delta_W*concentrations[3] - c_W*concentrations[3]**2
    return numpy.array([dG, dH, dD, dW])


# Weights in kg
H_weight = 4
D_weight = 200
W_weight = 40
weight_array = numpy.array([H_weight, D_weight, W_weight])

alpha_G = 0.2
e_H = 0.06
K_GH = 0.1
e_D = 0.06
K_GD = 0.1

alpha_H = 0.016
K_HW = 0.5
delta_H = 1.6e-4
c_H = 0.04
p_H = 0.05

alpha_D = 2.7e-3
K_DW = 1
delta_D = 2.7e-4
c_D = 0.003
p_D = 0.05

alpha_W = 0.068
delta_W = 3.4e-4
c_W = 0.03

# %% Simulate system deterministically with biomasses

time_grid = numpy.linspace(0, 15, 100)*365

# Hares and wolves
initial_concentrations = [1, 0.2, 0, 0.05]
time_array, concentration_matrix = integrate_system(
    species_derivatives, initial_concentrations=initial_concentrations, time_grid=time_grid)
figure = plot_concentrations(time_array/365, concentration_matrix, species_names=species_names,
                             filename="Hares wolves/Hares wolves biomass.svg", time_unit="years", colors=species_colors)

# Deer and wolves
initial_concentrations = [1, 0, 0.4, 0.02]
time_array, concentration_matrix = integrate_system(
    species_derivatives, initial_concentrations=initial_concentrations, time_grid=time_grid)
figure = plot_concentrations(time_array/365, concentration_matrix, species_names=species_names,
                             filename="Deer wolves/Deer wolves biomass.svg", time_unit="years", colors=species_colors)

# All species
initial_concentrations = [1, 0.2, 0.2, 0.05]
time_array, concentration_matrix = integrate_system(
    species_derivatives, initial_concentrations=initial_concentrations, time_grid=time_grid)
figure = plot_concentrations(time_array/365, concentration_matrix, species_names=species_names,
                             filename="All species/All species biomass.svg", time_unit="years", colors=species_colors)

# %% Simulate system deterministically with numbers

plot_scales = numpy.array([90, 1, 1, 1])


def mass_to_numbers(concentrations, total_weight=5000):
    """
    Converts concentrations to numbers
    """
    return numpy.array(concentrations) * total_weight / numpy.array([total_weight, H_weight, D_weight, W_weight])


def scale_concentration_matrix(concentration_matrix, scaling_matrix):
    """
    Scale concentration matrix by factors provided by the scaling matrix.
    """
    scaling_matrix = numpy.tile(
        plot_scales, (concentration_matrix.shape[1], 1)).T
    concentration_matrix *= scaling_matrix
    return


def integrate_numbers(initial_concentrations, total_weight=5000, time_span=[0, 5*365]):
    """
    Integrate system with biomasses converted to numbers
    """

    def species_derivatives_numbers(time, concentrations):
        dG = alpha_G*concentrations[0]*(1-concentrations[0]) - e_H*(H_weight/total_weight)*concentrations[1]*(concentrations[0]/(
            concentrations[0]+K_GH)) - e_D*(D_weight/total_weight)*concentrations[2]*(concentrations[0]/(concentrations[0]+K_GD))
        dH = alpha_H*concentrations[1]*concentrations[0]/(concentrations[0]+K_GH) - p_H*W_weight/H_weight*concentrations[3]*(concentrations[1]/(
            concentrations[1]+K_HW*(total_weight/H_weight))) - delta_H*concentrations[1] - c_H*(H_weight/total_weight)*concentrations[1]**2
        dD = alpha_D*concentrations[2]*concentrations[0]/(concentrations[0]+K_GD) - p_D*W_weight/D_weight*concentrations[3]*(concentrations[2]/(
            concentrations[2]+K_DW*(total_weight/D_weight))) - delta_D*concentrations[2] - c_D*(D_weight/total_weight)*concentrations[2]**2
        dW = alpha_W*concentrations[3]*(p_H*concentrations[1]/(concentrations[1]+K_HW*(total_weight/H_weight))+p_D*concentrations[2]/(
            concentrations[2]+K_DW*(total_weight/D_weight))) - delta_W*concentrations[3] - c_W*(W_weight/total_weight)*concentrations[3]**2
        return numpy.array([dG, dH, dD, dW])

    time_grid = numpy.linspace(*time_span, 100)
    return integrate_system(species_derivatives_numbers, initial_concentrations=initial_concentrations, time_grid=time_grid)


# Hares and wolves
total_weight = 10000
initial_concentrations = mass_to_numbers([1, 0.2, 0, 0.05], total_weight)
time_array, concentration_matrix = integrate_numbers(
    initial_concentrations, time_span=[0, 15*365], total_weight=total_weight)
scale_concentration_matrix(concentration_matrix, plot_scales)
figure = plot_concentrations(time_array/365, concentration_matrix, species_names=species_names,
                             filename="Hares wolves/Hares wolves numbers.svg", time_unit="years", colors=species_colors, figsize=(6, 5))

# Deer and wolves
total_weight = 25000
initial_concentrations = mass_to_numbers([1, 0, 0.4, 0.02], total_weight)
time_array, concentration_matrix = integrate_numbers(
    initial_concentrations, time_span=[0, 15*365], total_weight=total_weight)
scale_concentration_matrix(concentration_matrix, plot_scales)
figure = plot_concentrations(time_array/365, concentration_matrix, species_names=species_names,
                             filename="Deer wolves/Deer wolves numbers.svg", time_unit="years", colors=species_colors, figsize=(6, 5))

# All species
total_weight = 5000
initial_concentrations = mass_to_numbers([1, 0.2, 0.2, 0.05], total_weight)
time_array, concentration_matrix = integrate_numbers(
    initial_concentrations=initial_concentrations, time_span=[0, 15*365], total_weight=total_weight)
scale_concentration_matrix(concentration_matrix, plot_scales)
figure = plot_concentrations(time_array/365, concentration_matrix, species_names=species_names,
                             filename="All species/All species numbers.svg", time_unit="years", colors=species_colors, figsize=(6, 5))

# %% Functions for plotting in the spatial model

# Animals included in the model
animal_list = ["Hares", "Deer", "Wolves"]
animals = len(animal_list)
prey_list = [0, 1]
predator_list = [2]
animal_markers = ["o", "^", "v"]


def initialize_grid(grid):
    """
    Initialize plot of the animal grid.
    """

    grid_size = len(grid)

    # Create plot
    figure, axis = pyplot.subplots()

    # Show grass concentration in the background
    image = axis.imshow(grid[:, :, 0].T, cmap="YlGn", vmin=0, vmax=1)

    # List of scatter plots with one plot per species
    scatter_list = []

    # Iterate over all species
    for species_index, color in enumerate(species_colors[1:]):
        # Make a scatter plot of species positions
        positions = numpy.argwhere(grid[:, :, species_index+1])
        scatter_list.append(axis.scatter(positions[:, 0], positions[:, 1], color=color, edgecolor="black",
                                         s=50*30/grid_size, marker=animal_markers[species_index],
                                         label=species_names[species_index+1]))

    # Format plot
    fontsize = 15
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.legend(fontsize=fontsize, frameon=False, loc="upper left", bbox_to_anchor=(0, 1.13, 1, 0), ncol=3, mode="expand",
                borderaxespad=0)
    return figure, image, scatter_list


def update_grid(grid, figure, image, scatter_list, t):
    """
    Update plot of the animal grid.
    """

    axis = figure.get_axes()[0]

    # Set axis title
    axis.set_title(f"$t={t:.2f}$ days")

    # Draw grass
    image.set_data(grid[:, :, 0].T)
    # Draw animals
    for species_index, scatter_plot in enumerate(scatter_list):
        positions = numpy.argwhere(grid[:, :, species_index+1])
        scatter_plot.set_offsets(positions)

    # Update plot
    figure.canvas.draw()
    figure.canvas.flush_events()


def age_histogram(age_arrays, file_name=None):
    """
    Create a histogram showing animal ages at the end of the simulation.
    """
    fontsize = 15
    tick_label_size = 13

    # Iterate over all animals
    for animal_index, animal in enumerate(animal_list):
        # Get ages
        age_array = age_arrays[animal_index]
        # Skip empty age arrays
        if len(age_array) == 0:
            continue
        # Create histogram
        figure, axis = pyplot.subplots()
        axis.hist(age_array, color=species_colors[animal_index+1], bins=50)
        axis.set_title(f"{animal} age distribution", fontsize=fontsize)
        axis.tick_params(labelsize=tick_label_size)
        axis.set_xlabel("Days", fontsize=fontsize)
        axis.set_ylabel("Number of animals", fontsize=fontsize)
        figure.tight_layout()
        # Save figure if filename is set
        if file_name != None:
            figure.savefig(f"{file_name} {animal} ages.svg")


def lifespan_histogram(death_ages, file_name):
    """
    Create a histogram showing lifespans of animals.
    """
    fontsize = 15
    tick_label_size = 13

    # Iterate over all animals
    for animal_index, animal in enumerate(animal_list):
        # Get lifespans
        age_array = death_ages[animal_index]
        # Skip animals with no registered lifespans
        if len(age_array) == 0:
            continue
        # Create histogram
        figure, axis = pyplot.subplots()
        axis.hist(age_array, color=species_colors[animal_index+1], bins=50)
        axis.set_title(f"{animal} lifespans", fontsize=fontsize)
        axis.tick_params(labelsize=tick_label_size)
        axis.set_xlabel("Days", fontsize=fontsize)
        axis.set_ylabel("Number of animals", fontsize=fontsize)
        figure.tight_layout()

        # Save figure if filename is set
        if file_name != None:
            figure.savefig(f"{file_name} {animal} lifespan.svg")


def death_plot(death_causes, file_name):
    """
    Create a bar chart showing causes of death for animals.
    """

    fontsize = 15
    tick_label_size = 13

    # List of possible causes of death
    categories = ["natural", "predation"]
    category_labels = [category.capitalize() for category in categories]

    # Iterate over all animals
    for animal_index, animal in enumerate(animal_list):
        # Get causes of death
        death_dict = death_causes[animal_index]
        deaths = [death_dict[category] for category in categories]
        # Skip animal of no deaths are registered
        if numpy.sum(deaths) == 0:
            continue

        # Create bar chart
        figure, axis = pyplot.subplots()
        axis.bar(category_labels, deaths, color=species_colors[animal_index+1])
        axis.set_title(animal, fontsize=fontsize)
        axis.tick_params(labelsize=tick_label_size)
        axis.set_xlabel("Cause of death", fontsize=fontsize)
        axis.set_ylabel("Number of animals", fontsize=fontsize)
        figure.tight_layout()

        # Save figure if filename is set
        if file_name != None:
            figure.savefig(f"{file_name} {animal} death causes.svg")


def create_animation(time_list, grid_list, file_name=None, fps=10, bitrate=10000, time_unit="days"):
    """
    Creates an animation from a list of time points and species grids.

    Parameters
    ----------
    time_list : array_like
        List of time points.
    grid_list : array_like
        List of grids.
    file_name : string, optional
        File name to use for saving animation. The default is None, meaning animation will not be saved.
    fps : float, optional
        Frames per second in animation. The default is 10.
    bitrate : float, optional
        Bitrate of animation. The default is 60000.
    time_unit : str, optional
        Unit for time points. The default is "days".

    Returns
    -------
    grid_animation : matplotlib.animation.FuncAnimation
        The created animation.

    """
    print("Creating animation")

    frames = len(time_list)

    # Initialize figure
    grid = grid_list[0]
    figure, image, scatter_list = initialize_grid(grid)

    # Function that is run to create each frame
    def update_plot(frame):
        grid = grid_list[frame]
        t = time_list[frame]
        axis = figure.get_axes()[0]
        axis.set_title(f"$t={t:.2f}$ days")
        # Draw grass
        image.set_data(grid[:, :, 0].T)
        # Draw animals
        for species_index, scatter_plot in enumerate(scatter_list):
            positions = numpy.argwhere(grid[:, :, species_index+1])
            scatter_plot.set_offsets(positions)
        return figure

    # Create animation
    grid_animation = animation.FuncAnimation(
        figure, update_plot, frames=range(0, frames), blit=False, interval=13)

    # Save animation if filename is set
    if file_name != None:
        grid_animation.save(f"{file_name} animation.mp4",
                            fps=fps, bitrate=bitrate)

    return grid_animation


def grid_simulation(starting_concentrations, time_grid=numpy.linspace(0, 365, 3650), time_range=[0, 365], dt=0.1, seed=None,
                    grid_length=50, total_weight=5000, moving_probabilities=numpy.array([1, 1, 1]),
                    plot_interval=10, file_name=None, age_factor=0):
    """
    Simulates animals randomly walking on a grid and interacting.

    Parameters
    ----------
    starting_concentrations : array_like
        Initial concentrations of species. For grass, the concentration is the proportion of the maximum.
        For animals, relative biomasses are used.
    time_range : list, optional
        List with a start and stop time for the time range to cover during the simulation. The default is [0, 365].
    dt : float, optional
        Time step during each iteration. The default is 0.1.
    seed : int, optional
        Seed for the random number generator. The default is None.
    grid_length : int, optional
        Length of the grid in number of cells. The default is 50.
    total_weight : float, optional
        Biomass in kg corresponding to a concentration of 1. The default is 5000.
    moving_probabilities : numpy.ndarray, optional
        Array of length (animals) with the average number of times animals move per day. The default is numpy.array([1, 1, 1]).
    plot_interval : int, optional
        Number of iterations between progress updates in the plot and console. The default is 10.
    file_name : string, optional
        File name to use when saving plots without the extension. The default is None, meaning no plots are saved.
    age_factor : float, optional
        How much the probability of death increases per day. The default is 0.

    Returns
    -------
    None

    """

    # Initialize random number generator
    rng = numpy.random.default_rng(seed)

    # Possible steps when moving
    possible_steps = [numpy.array([1, 0]), numpy.array(
        [-1, 0]), numpy.array([0, 1]), numpy.array([0, -1])]

    # Functions for calculating growth and consumption of grass
    grid_cells = grid_length ** 2

    def G_growth_function(grass_concentration):
        return alpha_G*grass_concentration*(1-grass_concentration)

    def H_G_consumption(concentrations):
        return e_H * (H_weight/total_weight*grid_cells) * concentrations[0] / (concentrations[0] + K_GH)

    def D_G_consumption(concentrations):
        return e_D * (D_weight/total_weight*grid_cells) * concentrations[0] / (concentrations[0] + K_GD)

    G_consumption_functions = [H_G_consumption,
                               D_G_consumption, lambda concentrations: 0]

    # Functions for calculating reproduction probabilities of herbivores
    def H_reproduction_probability(concentrations):
        return alpha_H * concentrations[0] / (concentrations[0] + K_GH)

    def D_reproduction_probability(concentrations):
        return alpha_D * concentrations[0] / (concentrations[0] + K_GD)

    growth_functions = [H_reproduction_probability,
                        D_reproduction_probability, lambda concentrations: 0]

    # Functions for calculating probabilities of death
    def H_death_probability(concentrations, age):
        return delta_H + c_H*(H_weight/total_weight*grid_cells)*(concentrations[1]-1) + age_factor * age

    def D_death_probability(concentrations, age):
        return delta_D + c_D*(D_weight/total_weight*grid_cells)*(concentrations[2]-1) + age_factor * age

    def W_death_probability(concentrations, age):
        return delta_W + c_W*(W_weight/total_weight*grid_cells)*(concentrations[3]-1)

    death_functions = [H_death_probability,
                       D_death_probability, W_death_probability]

    # Functions for calculating probabilities of catching prey
    def H_predation_probability(concentrations):
        return p_H*(W_weight/H_weight)*concentrations[1] / (concentrations[1] + K_HW*total_weight/H_weight/grid_cells)

    def D_predation_probability(concentrations):
        return p_D*(W_weight/D_weight)*concentrations[2] / (concentrations[2] + K_DW*total_weight/D_weight/grid_cells)

    predation_probability_functions = [
        [H_predation_probability, D_predation_probability]]

    # Probabilities of reproducing after catching a prey
    H_predation_growth_probability = alpha_W * (H_weight / W_weight)
    D_predation_growth_probability = alpha_W * (D_weight / W_weight)
    predator_growth_probabilities = [
        [H_predation_growth_probability, D_predation_growth_probability]]

    starting_numbers = numpy.round(
        total_weight*starting_concentrations/weight_array).astype(int)

    # Grid storing concentrations in each cell
    grid = numpy.array([[([1.0] + [0 for animal in animal_list])
                       for y in range(grid_length)] for x in range(grid_length)])
    # Grid containing animal dicts. Each grid cell contains a list of length len(animal_list) with one list per species of all individuals in that location
    animal_grid = [[[[] for animal in animal_list]
                    for y in range(grid_length)] for x in range(grid_length)]

    # Add animals to grid and animal_grid
    for species_index, individuals in enumerate(starting_numbers):
        for individual_index in range(individuals):
            position = numpy.floor(rng.random(2)*grid_length).astype(int)
            grid[*position, species_index+1] += 1
            new_animal = {"position": position, "age": 0}
            animal_grid[position[0]][position[1]
                                     ][species_index].append(new_animal)

    print("Grid initialized")

    # Initialize plot
    figure, image, scatter_list = initialize_grid(grid)

    time_grid = numpy.arange(*time_range, dt)
    concentration_matrix = []

    # Create data structures for storing results
    death_ages = [[] for animal in animal_list]
    death_causes = [{"natural": 0, "predation": 0} for animal in animal_list]
    grid_list = []
    time_list = []

    # Indices to update the plot at
    plot_indices = numpy.zeros(len(time_grid), dtype=bool)
    plot_indices[::plot_interval] = True
    plot_indices[-1] = True

    for t, plot in zip(time_grid, plot_indices):

        # Grow grass, and prevent the length from becoming 0 or lower
        grid[:, :, 0] = numpy.clip(
            grid[:, :, 0]+G_growth_function(grid[:, :, 0])*dt, a_min=1e-2, a_max=numpy.inf)
        new_animal_grid = [[[[] for animal in animal_list]
                            for y in range(grid_length)] for x in range(grid_length)]

        # Iterate over the entire grid
        for x in range(grid_length):
            for y in range(grid_length):
                concentrations = grid[x, y]
                animal_lists = animal_grid[x][y]

                # Perform calculations for each species
                for animal_index in range(animals):
                    for grid_index, animal in enumerate(animal_lists[animal_index]):
                        # Eat grass
                        grid[x, y, 0] -= G_consumption_functions[animal_index](
                            concentrations) * dt

                        # Die
                        death_probability = death_functions[animal_index](
                            concentrations, animal["age"]) * dt
                        if death_probability > rng.random():
                            grid[x, y, animal_index+1] -= 1
                            death_ages[animal_index].append(animal["age"])
                            death_causes[animal_index]["natural"] += 1
                            continue

                        # Reproduce
                        reproduction_probability = growth_functions[animal_index](
                            concentrations) * dt
                        if reproduction_probability > rng.random():
                            offspring_position = numpy.floor(
                                rng.random(2)*grid_length).astype(int)
                            grid[*offspring_position, animal_index+1] += 1
                            new_animal = {
                                "position": offspring_position, "age": 0}
                            new_animal_grid[offspring_position[0]][offspring_position[1]][animal_index].append(
                                new_animal)

                        # Move
                        if moving_probabilities[animal_index]*dt > rng.random():
                            step = rng.choice(possible_steps)
                            new_position = numpy.clip(numpy.array(
                                [x, y])+numpy.array(step), a_min=0, a_max=grid_length-1)
                            grid[x, y, animal_index+1] -= 1
                            grid[*new_position, animal_index+1] += 1
                            animal["position"] = new_position

                        animal_position = animal["position"]
                        animal["age"] += dt
                        new_animal_grid[animal_position[0]][animal_position[1]][animal_index].append(
                            animal)

        # Predation
        for x in range(grid_length):
            for y in range(grid_length):
                cell_animals = new_animal_grid[x][y].copy()
                for predator_index, predator_animal_index in enumerate(predator_list):
                    # Iterate over all predators in the cell
                    for predator in cell_animals[predator_animal_index]:
                        # For each prey type, check if the predator eats it
                        for prey_index, prey_animal_index in enumerate(prey_list):
                            predation_probability = predation_probability_functions[predator_index][prey_index](
                                grid[x, y]) * dt
                            if predation_probability > rng.random():
                                # Eat prey
                                grid[x, y, prey_animal_index+1] -= 1
                                prey_age = new_animal_grid[x][y][prey_animal_index][-1]["age"]
                                new_animal_grid[x][y][prey_animal_index].pop()
                                death_ages[prey_animal_index].append(prey_age)
                                death_causes[prey_animal_index]["predation"] += 1
                                # Have a chance of reproducing
                                reproduction_probability = predator_growth_probabilities[
                                    predator_index][prey_index]
                                if reproduction_probability > rng.random():
                                    # Reproduce
                                    offspring_position = numpy.floor(
                                        rng.random(2)*grid_length).astype(int)
                                    grid[*offspring_position,
                                         predator_animal_index+1] += 1
                                    new_predator = {
                                        "position": offspring_position, "age": 0}
                                    new_animal_grid[offspring_position[0]][offspring_position[1]][predator_animal_index].append(
                                        new_predator)

        # Update animal grid
        animal_grid = new_animal_grid

        # Update concentrations
        concentrations = numpy.sum(grid, axis=(0, 1))
        concentration_matrix.append(concentrations)

        # Show grid for some time points
        if plot:
            update_grid(grid, figure, image, scatter_list, t)
            grid_list.append(grid.copy())
            time_list.append(t)
            print(f"\rt = {t:.2f}", end="")

    print()

    # Rescale and transpose concentration matrix
    scales = numpy.array([0.1, 1, 1, 1])
    concentration_matrix = numpy.array(concentration_matrix)*scales
    concentration_matrix = concentration_matrix.T
    # Plot concentrations over time
    plot_concentrations(time_grid, concentration_matrix, species_names=species_names,
                        time_unit="days", colors=species_colors, filename=file_name+" concentrations.svg")

    # Get age of living animals
    age_arrays = [[] for animal_index in range(animals)]
    for x in range(grid_length):
        for y in range(grid_length):
            for animal_index, species_list in enumerate(animal_grid[x][y]):
                for animal in species_list:
                    age_arrays[animal_index].append(animal["age"])

    # Plot ages, lifespans and causes of death
    age_histogram(age_arrays, file_name)
    lifespan_histogram(death_ages, file_name)
    death_plot(death_causes, file_name)

    # Create animation
    create_animation(time_list, grid_list, file_name=file_name)

    return

# %% Hares and wolves no aging
years=15
grid_simulation(numpy.array([0.2, 0, 0.05]), seed=0, dt=0.1, time_range=[0, years*365], grid_length=30, plot_interval=10,
                file_name="Hares wolves/Hares wolves no aging", moving_probabilities=numpy.array([5, 5, 5]), total_weight=10000)

# %% Deer wolves no aging
years=15
grid_simulation(numpy.array([0, 0.4, 0.02]), seed=0, total_weight=50000, dt=0.1, grid_length=30,
                time_range=[0, years*365], file_name="Deer wolves/Deer wolves no aging", plot_interval=10, moving_probabilities=numpy.array([5, 5, 5]))

# %% All species no aging
years=15
grid_simulation(numpy.array([0.2, 0.2, 0.05]), seed=0, time_range=[0, years*365], dt=0.1, plot_interval=10,
                grid_length=30, total_weight=5000, file_name="All species/All species no aging", moving_probabilities=numpy.array([5, 5, 5]))

# %% Hares wolves aging
years = 15
grid_simulation(numpy.array([0.2, 0, 0.05]), seed=7, time_range=[0, years*365], dt=0.1, plot_interval=10,
                grid_length=30, file_name="Hares wolves/Hares wolves aging", moving_probabilities=numpy.array([5, 5, 5]), total_weight=10000,
                age_factor=4e-6)

# %% Deer wolves aging
years = 15
grid_simulation(numpy.array([0, 0.4, 0.02]), seed=11, time_range=[0, years*365], dt=0.1, plot_interval=10,
                grid_length=30, file_name="Deer wolves/Deer wolves aging",
                moving_probabilities=numpy.array([5, 5, 5]), total_weight=50000, age_factor=4e-6)

# %% All species aging
years = 15
grid_simulation(numpy.array([0.2, 0.2, 0.05]), seed=8, time_range=[0, years*365], dt=0.1, plot_interval=10,
                grid_length=30, total_weight=5000, file_name="All species/All species aging", moving_probabilities=numpy.array([5, 5, 5]),
                age_factor=4e-6)
