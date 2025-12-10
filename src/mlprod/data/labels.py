import numpy as np

from mlprod.api.requests import LocationData, UserData
from mlprod.data.configs import UserConfig


class UserLabeller:
    """This class is used to create a labeller that aims to copy the decisional process of a real user."""

    def __init__(
        self,
        config: UserConfig,
    ) -> None:
        """Creates a new labeller object for given users' settings.

        The settings values are used as following:

        * budget_tolerance:
            Percentage of over budget that can be considered acceptable.
            Example: 0.1 with a budget of 1000 means that an offer of 1100 is acceptable

        * facilities_tolerance:
            Percentage of the user requested facilities that are met by the
            location. User desire can be manipulated using ``weight_*`` parameters
            Example:
                User wants 4 facilities, location offer 3, this has a 0.75 achieved score.
                With a threshold of 0.8, the location is rejected.

        * weight_*:
            These parameters manage the desired level of each facility of a location.
            Weight can be between 0 and 1. With a score less than 1, means that the
            desire is weak and that the absence of the facility (if selected in
            UserData object) is not an huge issue.

        * weight_score_*:
            These parameters control the weight of the final score. The three
            categories are:
                - weight_score_facilities (what the location offers)
                - weight_score_environment (what the location surrounding offers)
                - weight_score_users (what other users reviews says on the location)

        * exploration_tolerance:
            If above this threshold, chose randomly one locations else choose the best one.

        * ignore_tolerance:
            If above this threshold, ignore all results (no choice made).

        :param config:
            User settings read from a configuration file.
        """
        self.config = config

    def score(self, user: UserData, location: LocationData) -> float:
        """This functions simulates the decision process of the user.

        This process can be manipulated with the weight parameters of the input data object. At
        the end, the location is assigned a scores greater than 0 where 0 means a bad location.

        :param user:
            The user's parameters to use as scorer.

        :param location:
            The description of the location to score based on the user parameters.
        """
        # check for budget:
        user_budget = user.budget + self.config.budget_tolerance * user.budget

        if location.price > user_budget:
            # over budget, reject
            return 0

        # check for facilities
        facilities_score_desired = 0
        facilities_score_achieved = 0
        environment_score_desired = 0
        environment_score_achieved = 0

        if user.spa:
            facilities_score_desired += self.config.weight_spa
            facilities_score_achieved += user.spa == location.has_spa
        if user.pool:
            facilities_score_desired += self.config.weight_pool
            facilities_score_achieved += user.pool == location.has_pool
        if user.pet_friendly:
            facilities_score_desired += self.config.weight_pet
            facilities_score_achieved += user.pet_friendly == location.animals
        if user.lake:
            environment_score_desired += self.config.weight_lake
            environment_score_achieved += user.lake == location.near_lake
        if user.mountain:
            environment_score_desired += self.config.weight_mountains
            environment_score_achieved += user.mountain == location.near_mountains
        if user.sport:
            facilities_score_desired += self.config.weight_sport
            facilities_score_achieved += user.sport == location.has_sport
            environment_score_desired += self.config.weight_sport
            environment_score_achieved += user.sport == location.has_sport

        if (
            facilities_score_desired > 0
            and facilities_score_achieved / facilities_score_desired
            < self.config.facilities_tolerance
        ):
            # not enough facilities
            return 0

        if (
            environment_score_desired > 0
            and environment_score_achieved / environment_score_desired
            < self.config.environment_tolerance
        ):
            # not enough facilities
            return 0

        if user.children_num and location.family_rating < self.config.family_tolerance:
            # location not right for families
            return 0

        # location is acceptable, assign score
        score = (
            self.config.weight_score_facilities
            * (
                location.family_rating
                + location.service_rating
                + facilities_score_achieved
            )
            + self.config.weight_score_environment
            * (
                location.outdoor_rating
                + location.food_rating
                + location.leisure_rating
                + environment_score_achieved
            )
            + self.config.weight_score_users * location.user_score
        )

        return score

    def __call__(
        self, r: np.random.Generator, user: UserData, locations: list[LocationData]
    ) -> np.ndarray:
        """This function will process a list of possible locations assigning them a score.

        Then a explotation/exploration mechanism choose which location has been choosed assignign a "1" label to the return label vector.

        :param r:
            Random generator object to use.

        :param user:
            The user's parameters to use in the scorer.

        :param locations:
            A list of locations parameters to score for the given user.
        """
        scores = np.array([self.score(user, loc) for loc in locations])
        labels = np.zeros(scores.shape)

        if scores.sum() == 0:
            # no suitable locations
            return labels

        n = scores.shape[0]
        p = scores / scores.sum()

        # Choosen location
        if r.uniform() > self.config.ignore_tolerance:
            # ignore results
            return labels

        if r.uniform() > self.config.exploration_tolerance:
            # exploit
            labels[np.argmax(scores)] = 1
        else:
            # explore
            c = r.choice(n, p=p)
            labels[c] = 1

        return labels
