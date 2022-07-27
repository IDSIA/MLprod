import numpy as np

from api.requests import LocationData, UserData

class UserLabeller:

    def __init__(
        self, 
        budget_tolerance: float = 0.0, 
        facilities_tolerance: float = 0.0, 
        family_tolerance: float= 1.0, 
        weight_spa: float=1.0, 
        weight_pool: float=1.0, 
        weight_pet: float=1.0, 
        weight_lake: float=1.0, 
        weight_mouintains: float=1.0, 
        weight_sport: float=1.0, 
        weight_score_facilities: float=1.0, 
        weight_score_envirnonment: float=1.0, 
        weight_score_users: float=1.0
    ) -> None:

        """
        :param budget_tolerance:
          Percentage of over budget that can be considered acceptable.
          Example:
            0.1 with a budget of 1000 means that an offer of 1100 is acceptable 

        :param facilities_tolerance:
          Percentage of the user requested facilities that are met by the 
          location. User desire can be manipulated using ``weight_*`` parameters
          Example:
            User wants 4 facilities, location offer 3, this has a 0.75 achieved score. With a threshold of 0.8, the location is rejected.

        :param weight_*:
          These parameters manage the desired level of each facility of a location. 
          Weight can be between 0 and 1. With a score less than 1, means that the 
          desire is weak and that the absence of the facility (if selected in 
          UserData object) is not an huge issue.

        :param weight_score_*:
          These parameters control the weight of the final score. The three 
          categories are: 
            - weight_score_facilities (what the location offers)
            - weight_score_envirnonment (what the location surroundign offers)
            - weight_score_users (what other users reviews says on the location)
        """
        self.budget_tolerance = budget_tolerance
        self.facilities_tolerance = facilities_tolerance
        self.family_tolerance = family_tolerance

        self.weight_spa = weight_spa
        self.weight_pool = weight_pool
        self.weight_pet = weight_pet
        self.weight_lake = weight_lake
        self.weight_mouintains = weight_mouintains
        self.weight_sport = weight_sport

        self.weight_facilities = weight_score_facilities
        self.weight_envirnonment = weight_score_envirnonment
        self.weight_users = weight_score_users

    def score(self, user: UserData, location: LocationData) -> float:
        """
        This functions simulates the decision process of the user. This process
        can be manipulated with the weight parameters of the input data object. At
        the end, the location is assigned a scores greather than 0 where 0 menas a
        bad location.
        """

        # check for budget:
        user_budget = user.budget + self.budget_tolerance * user.budget

        if location.price > user_budget:
            # over budget, reject
            return 0

        # check for facilities
        facilities_score_desired = 0
        facilities_score_achieved = 0

        if user.spa:
            facilities_score_desired += self.weight_spa
            facilities_score_achieved += user.spa == location.spa
        if user.pool:
            facilities_score_desired += self.weight_pool
            facilities_score_achieved += user.pool == location.pool
        if user.pet_friendly:
            facilities_score_desired += self.weight_pet
            facilities_score_achieved += user.pet_friendly == location.animals
        if user.lakes:
            facilities_score_desired += self.weight_lake
            facilities_score_achieved += user.lakes == location.lake
        if user.mountains:
            facilities_score_desired += self.weight_mouintains
            facilities_score_achieved += user.mountains == location.mountain
        if user.sport:
            facilities_score_desired += self.weight_sport
            facilities_score_achieved += user.sport == location.sport

        if facilities_score_desired > 0 and \
            facilities_score_achieved / facilities_score_desired < self.facilities_tolerance:
            # not enough facilities
            return 0

        if user.children and location.family_rating < self.family_tolerance:
            # location not right for families
            return 0

        # location is acceptable, assign score
        score = (
            self.weight_facilities * (
                location.family_rating + 
                location.service_rating + 
                facilities_score_achieved
            ) + 
            self.weight_envirnonment * (
                location.outdoor_rating + 
                location.food_rating + 
                location.leisure_rating
            ) +
            self.weight_users * location.user_score            
        )

        return score

    def __call__(self, r: np.random.Generator, user: UserData, locations: list[LocationData]) -> int:
        """
        This function will process a list of possible locations assigning them a
        score. Then a explotation/exploration mechanism choose which location has 
        been choosed assignign a "1" label to the return label vector.
        """

        scores = np.array([self.score(user, loc) for loc in locations])
        labels = np.zeros(scores.shape)

        if scores.sum() == 0:
            # no suitable locations
            return labels

        n = scores.shape[0]
        p = scores / scores.sum()

        # Choosen location
        if r.uniform(0, 1) > 0.5:
            # exploit
            labels[np.argmax(scores)] = 1
        else:
            # explore
            c = r.choice(n, p=p)
            labels[c] = 1

        return labels
