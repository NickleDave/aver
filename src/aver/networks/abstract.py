import nengo


class ActiveVision(nengo.Network):
    """Abstract class that represents an active vision model with a Nengo network.

    This abstract class asserts that an active vision model must address the following
    four questions (taken from [1]_):
    1. how is the decision made when to terminate one fixation and move the gaze?
    2. how is the decision made where to move the gaze to sample the visual environment?
    3. what information is taken in during a fixation?
    4. how is information integrated across fixations?

    So, each question is represented by a static method of this abstract class. To define
    an active vision model, simply sub-class the ActiveVisionNetwork, and then implement
    those static methods. For an example, see aver.networks.example

    Methods
    -------
    WhenNet: network that determines when to terminate one fixation and move the gaze
    WhereNet: network that determines where to move the gaze
    WhatNet: network that determines what information is taken in during a fixation
    HowNet: Network that determines how information is integrated across fixations

    References
    ----------
    .. [1] Findlay, John M., and Iain D. Gilchrist.
    *Active vision: The psychology of looking and seeing.*
    No. 37. Oxford University Press, 2003.
    https://www.worldcat.org/title/active-vision-the-psychology-of-looking-and-seeing/
    """
    @staticmethod
    def WhenNet():
        """Network that determines when to terminate one fixation and move the gaze"""
        raise NotImplementedError

    @staticmethod
    def WhereNet():
        """Network that determines where to move the gaze"""
        raise NotImplementedError

    @staticmethod
    def WhatNet():
        """Network that determines what information is taken in during a fixation"""
        raise NotImplementedError

    @staticmethod
    def HowNet():
        """Network that determines how information is integrated across fixations"""
        raise NotImplementedError

    def __init__(self, net=None, label=None, seed=None, add_to_container=None):
        super(ActiveVisionNetwork, self).__init__(label, seed, add_to_container)

        if net:
            self.net = net
        else:
            self.net = nengo.Network()

        with self.net:
            when_net = self.WhenNet()
            where_net = self.WhereNet()
            what_net = self.WhatNet()
            how_net = self.HowNet()
