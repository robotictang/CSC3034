# Copyright Author: Dr Tang Tiong Yew
import random
import matplotlib.pyplot as plt

class Particle:
  def __init__(self, position = [0, 0], velocity = [0, 0]):
    self.position = position
    self.velocity = velocity
    self.best_position = position
    self.position_list = [[it for it in position]]
    self.velocity_list = [[it for it in velocity]]
    self.best_position_list = []

  def update_personal_best(self):
    # 1. calculate the fitnesses of the best_position and the particle's current position
    best_pos_fitness = fit_fcn(self.best_position)
    current_pos_fitness = fit_fcn(self.position)
    # 2. compare the fitnesses and determine if the current position is better than the best_position
    if current_pos_fitness < best_pos_fitness:
      # 3. update if necessary
      self.best_position = self.position
    # 4. no return statement is required
    self.best_position_list.append([it for it in self.best_position])

  def update_velocity(self, alpha, beta, glob_best_pos):
    # alpha is a list of two values. we will access alpha_1 and alpha_2 by alpha[0] and alpha[1] respectively. This also applies to beta.
    # the current position, current velocity, and personal best position of the particle can be accessed by self.position, self.velocity, and self.best_position
    # assign the particle's velocity with the updated velocity
    self.velocity[0] = 0.5*self.velocity[0] + alpha[0] * beta[0] * (self.best_position[0] - self.position[0]) + alpha[1] * beta[1] * (glob_best_pos[0] - self.position[0])
    self.velocity[1] = 0.5*self.velocity[1] + alpha[0] * beta[0] * (self.best_position[1] - self.position[1]) + alpha[1] * beta[1] * (glob_best_pos[1] - self.position[1])
    # self.velocity = alpha[0] * beta[0] * (self.best_position - self.position) + alpha[1] * beta[1] * (glob_best_pos - self.position)
    self.velocity_list.append([it for it in self.velocity])

  def update_position(self, position_limits):
    self.position[0] = self.position[0] + self.velocity[0]
    self.position[1] = self.position[1] + self.velocity[1]
    # how should you solve the problem of the position (x) going out of the limits
    self.position[0] = max(min(self.position[0], position_limits[0][1]), position_limits[0][0])
    self.position[1] = max(min(self.position[1], position_limits[1][1]), position_limits[1][0])
    self.position_list.append([it for it in self.position])
    
def fit_fcn(position):
  fitness = (position[0] ** 2) * (position[1] ** 2)
  return fitness

def initialise_particles(n_ptc, position_limits):
  # position_limits is a list of two values. The first value is the lower boundary and the second value is the upper boundary.
  particles = [Particle([random.random() * (position_limits[0][1] - position_limits[0][0]) + position_limits[0][0], random.random() * (position_limits[1][1] - position_limits[1][0]) + position_limits[1][0]]) for _ in range(n_ptc)]
  return particles

def compareFitness(pos1, pos2):
  # 1. calculate the fitness of pos1 and pos2
  fitness1 = fit_fcn(pos1)
  fitness2 = fit_fcn(pos2)
  # 2. compare to determine the better position
  if fitness1 < fitness2:
    betterpos = pos1
  else:
    betterpos = pos2
  return betterpos

def calc_avg_fit_diff(particles):
  # 1. calculate mean fitness of all particles
  avg_fit = sum([fit_fcn(p.position) for p in particles]) / len(particles)
  # 2. calculate the difference between the mean fitness and the fitness of each particle
  fit_diff = [abs(fit_fcn(p.position) - avg_fit) for p in particles]
  # 3. calculate the average of the differences obtained from step 2
  avg_fit_diff = sum(fit_diff) / len(particles)
  return avg_fit_diff

def calc_avg_pos_diff(particles):
  # 1. calculate mean position of all particles
  avg_pos = [sum([p.position[0] for p in particles]) / len(particles), sum([p.position[1] for p in particles]) / len(particles)]
  # 2. calculate the difference between the mean position and the position of each particle
  pos_diff = [(p.position[0] - avg_pos[0])**2 + (p.position[1] - avg_pos[1])**2 for p in particles]
  # 3. calculate the average of the differences obtained from step 2
  avg_pos_diff = sum(pos_diff) / len(particles)
  return avg_pos_diff

if __name__ == '__main__':
  # parameter initialisation
  alpha = [0.05, 0.1]
  n_particle = 10
  global_best_position = None
  global_best_position_list = []
  position_limits = [[-100, 100],[-100, 100]]
  fitness_range = [None, None]
  # termination threshold
  iteration = 0
  max_iter = 200
  min_avg_fit_diff = 0.1
  min_avg_pos_diff = 0.1
  # initialise particles
  particles = initialise_particles(n_particle, position_limits)
  for p in particles:
    this_fit = fit_fcn(p.position)
    if fitness_range[0] == None:
      fitness_range = [this_fit, this_fit]
    else:
      if this_fit < fitness_range[0]:
        fitness_range[0] = this_fit
      if this_fit > fitness_range[1]:
        fitness_range[1] = this_fit
  space_ax = plt.axes()
  colorgraph = space_ax.scatter([p.position[0] for p in particles], [p.position[1] for p in particles], c=[fit_fcn(p.position) for p in particles], vmin=fitness_range[0], vmax=fitness_range[1])
  space_ax.set_xlim(*position_limits[0])
  space_ax.set_ylim(*position_limits[1])
  space_ax.set_title("Position of particles in iteration {}".format(iteration))
  space_ax.set_xlabel("Position $x$")
  space_ax.set_ylabel("Position $y$")
  colorbar = plt.colorbar(colorgraph)
  while (iteration < max_iter and calc_avg_fit_diff(particles) > min_avg_fit_diff and calc_avg_pos_diff(particles) > min_avg_pos_diff): # how should you define the termination criteria here?
    print(iteration, [[round(p.position[0],2), round(p.position[1],2)] for p in particles])
    space_ax.cla()
    colorgraph = space_ax.scatter([p.position[0] for p in particles], [p.position[1] for p in particles], c=[fit_fcn(p.position) for p in particles], vmin=fitness_range[0], vmax=fitness_range[1])
    space_ax.set_xlim(*position_limits[0])
    space_ax.set_ylim(*position_limits[1])
    space_ax.set_title("Position of particles in iteration {}".format(iteration))
    space_ax.set_xlabel("Position $x$")
    space_ax.set_ylabel("Position $y$")
    colorbar.update_normal(colorgraph)
    plt.pause(0.1) # pause the program for 0.5 second; if graph changes too quickly, increase this value; you can also speed up the process by decreasing this value
    for particle in particles:
      # update personal best
      particle.update_personal_best()
      # update global best
      if global_best_position == None:
        global_best_position = particle.position
      else:
        global_best_position = compareFitness(global_best_position, particle.position)
    global_best_position_list.append(global_best_position) # take note on the indentation
    fitness_range[0] = fit_fcn(global_best_position)
    # generate beta randomly for current iteration
    beta = [random.random(), random.random()]
    for particle in particles:
      # update velocity
      particle.update_velocity(alpha, beta, global_best_position)
      # update position
      particle.update_position(position_limits)
    iteration += 1
    for p in particles:
        this_fit = fit_fcn(p.position)
        if this_fit > fitness_range[1]:
            fitness_range[1] = this_fit
  # display results
  print(iteration, [[round(p.position[0],2), round(p.position[1],2)] for p in particles])
  space_ax.cla()
  colorgraph = space_ax.scatter([p.position[0] for p in particles], [p.position[1] for p in particles], c=[fit_fcn(p.position) for p in particles], vmin=fitness_range[0], vmax=fitness_range[1])
  space_ax.set_xlim(*position_limits[0])
  space_ax.set_ylim(*position_limits[1])
  space_ax.set_title("Position of particles in iteration {}".format(iteration))
  space_ax.set_xlabel("Position $x$")
  space_ax.set_ylabel("Position $y$")
  colorbar.update_normal(colorgraph)

  plt.show()
  

[pos_fig, position_axes] = plt.subplots(6,1,sharex=True)
position_axes[0].set_title("Position $x$ of each particle")
position_axes[1].set_title("Boxplot of position $x$ at each iteration")
position_axes[2].set_title("Position $y$ of each particle")
position_axes[3].set_title("Boxplot of position $y$ at each iteration")
position_axes[4].set_title("Fitness of each particle")
position_axes[5].set_title("Boxplot of fitness at each iteration")
position_axes[5].set_xlabel("Iteration")
[vel_fig, velocity_axes] = plt.subplots(4,1,sharex=True)
velocity_axes[0].set_title("Velocity for $x$ of each particle")
velocity_axes[1].set_title("Boxplot for velocity for $x$ at each iteration")
velocity_axes[2].set_title("Velocity for $y$ of each particle")
velocity_axes[3].set_title("Boxplot for velocity for $y$ at each iteration")
velocity_axes[3].set_xlabel("Iteration")
[p_best_fig, personal_best_axes] = plt.subplots(6,1,sharex=True)
personal_best_axes[0].set_title("Personal best position for $x$ of each particle")
personal_best_axes[1].set_title("Boxplot of personal best position for $x$ at each iteration")
personal_best_axes[2].set_title("Personal best position for $y$ of each particle")
personal_best_axes[3].set_title("Boxplot of personal best position for $y$ at each iteration")
personal_best_axes[4].set_title("Personal best fitness of each particle")
personal_best_axes[5].set_title("Boxplot of personal best fitness at each iteration")
personal_best_axes[5].set_xlabel("Iteration")
[g_best_fig, global_best_axes] = plt.subplots(3,1,sharex=True)
global_best_axes[0].set_title("Global best position for $x$")
global_best_axes[1].set_title("Global best position for $y$")
global_best_axes[2].set_title("Fitness for global best position")
global_best_axes[2].set_xlabel("Iteration")
for particle in particles:
  iteration_list = list(range(len(particle.position_list)))
  x_pos_list = [x for x,y in particle.position_list]
  y_pos_list = [y for x,y in particle.position_list]
  position_axes[0].plot(iteration_list, x_pos_list, '-o')
  position_axes[2].plot(iteration_list, y_pos_list, '-o')
  position_axes[4].plot(iteration_list, [fit_fcn(x) for x in particle.position_list], '-o')

  x_vel_list = [x for x,y in particle.velocity_list]
  y_vel_list = [y for x,y in particle.velocity_list]
  velocity_axes[0].plot(iteration_list, x_vel_list, '-o')
  velocity_axes[2].plot(iteration_list, y_vel_list, '-o')

  x_best_list = [x for x,y in particle.best_position_list]
  y_best_list = [x for x,y in particle.best_position_list]
  personal_best_axes[0].plot(iteration_list[:-1], x_best_list, '-o')
  personal_best_axes[2].plot(iteration_list[:-1], y_best_list, '-o')
  personal_best_axes[4].plot(iteration_list[:-1], [fit_fcn(x) for x in particle.best_position_list], '-o')

position_axes[1].boxplot([[p.position_list[i][0] for p in particles] for i in iteration_list], positions=iteration_list)
position_axes[3].boxplot([[p.position_list[i][1] for p in particles] for i in iteration_list], positions=iteration_list)
position_axes[5].boxplot([[fit_fcn(p.position_list[i]) for p in particles] for i in iteration_list], positions=iteration_list)

velocity_axes[1].boxplot([[p.velocity_list[i][0] for p in particles] for i in iteration_list], positions=iteration_list)
velocity_axes[3].boxplot([[p.velocity_list[i][1] for p in particles] for i in iteration_list], positions=iteration_list)

personal_best_axes[1].boxplot([[p.best_position_list[i][0] for p in particles] for i in iteration_list[:-1]], positions=iteration_list[:-1])
personal_best_axes[3].boxplot([[p.best_position_list[i][1] for p in particles] for i in iteration_list[:-1]], positions=iteration_list[:-1])
personal_best_axes[5].boxplot([[fit_fcn(p.best_position_list[i]) for p in particles] for i in iteration_list[:-1]], positions=iteration_list[:-1])

x_best_list = [x for x,y in global_best_position_list]
y_best_list = [x for x,y in global_best_position_list]
global_best_axes[0].plot(iteration_list[:-1], x_best_list, '-o')
global_best_axes[1].plot(iteration_list[:-1], y_best_list, '-o')
global_best_axes[2].plot(iteration_list[:-1], [fit_fcn(x) for x in global_best_position_list], '-o')

# plt.pause(0.1)
plt.show()
# input()