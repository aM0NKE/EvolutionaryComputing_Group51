from evoman.controller import Controller
import neat

class NEATController(Controller):
    """ 
        This class implements the NEAT controller.
    """

    def __init__(self):
        """
            Initializes the NEAT controller.
        """
        self.config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                'NEATConfig')

    def control(self, inputs, genome):
        """
            Calculates the actions to be taken by the sprite.
            Based on a Neural Network optimized with the NEAT algorithm.

            Args:
                inputs (list): The inputs to the neural network.
                genome (list): The genome to be evaluated.
        """
        # Create and activate neural network
        nnet = neat.nn.FeedForwardNetwork.create(genome, self.config)
        output = nnet.activate(inputs)

        # Takes decisions about sprite actions
        if output[0] > 0.5:
            left = 1
        else:
            left = 0

        if output[1] > 0.5:
            right = 1
        else:
            right = 0

        if output[2] > 0.5:
            jump = 1
        else:
            jump = 0

        if output[3] > 0.5:
            shoot = 1
        else:
            shoot = 0

        if output[4] > 0.5:
            release = 1
        else:
            release = 0

        return [left, right, jump, shoot, release]
