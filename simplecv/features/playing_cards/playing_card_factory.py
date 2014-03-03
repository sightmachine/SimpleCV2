import warnings

from simplecv.features.features import FeatureSet
from simplecv.features.playing_cards.playing_card import PlayingCard


class CardError(Exception):
    def __init__(self, card=None, message=None):
        self.card = card
        self.msg = message

    def __str__(self):
        return repr(self.msg)


class PlayingCardFactory():
    def __init__(self, parameter_dict=None):
        if parameter_dict is not None:
            self.parameterize(parameter_dict)

    def parameterize(self, parameter_dict):
        """
        Parameterize from a dictionary so we can optimize performance.
        """
        pass

    def process(self, img):
        """
        Process the image. Return a featureset with a single
        PlayingCard feature or None
        """
        # Can we find anything that looks like a card
        card = self._find_card_edges(img)
        if card is None:  # if we don't see it just bail
            warnings.warn("Could not find a card.")
            return None
        try:
            # extract the basic features and get color
            card = self._estimate_color(card)
            # okay, we got a color and some features
            # go ahead and estimate the suit
            card = self._estimate_suit(card)
            # Do we think this is a face card this
            # is an easier test
            is_face, card = self._is_face_card(card)
            if is_face:
                # if we are a face card get the face. This is had
                card = self._estimate_face_card(card)
            else:
                # otherwise get the rank
                # first pass is corners second
                # pass is the card body
                card = self._estimate_rank(card)
            # now go back do some sanity checks
            # and cleanup the features so it is not
            # too heavy
            card = self._refine_estimates(card)
        except CardError as ce:
            card = ce.card
            if card is not None:
                # maybe we got a joker or someone
                # is being a jackass and showing us the
                # back of the card.
                card = self._is_non_standard_card(card)
            warnings.warn(ce.msg)  # we may swallow this later
            # optionally we may want to log these to
            # see where we fail and why or do a parameter
            # adjustment and try again
        except:
            # this means we had an error somewhere
            # else maybe numpy
            print "Generic Error."
            return None
        return FeatureSet([card])

    def _preprocess(self, img):
        """
        Any image preprocessing options go here.
        """
        return img

    def _find_card_edges(self, img):
        """
        Try to find a card, if we do return a card feature
        otherwise return None
        """
        ppimg = self._preprocess(img)
        result = PlayingCard(img, img.width / 2, img.height / 2)
        # create the feature, hang any preprocessing
        # steps on the feature
        return result

    def _estimate_color(self, card):
        """
        Take in a card feature and determine the color.
        if we find a color return the feature otherwise
        throw.
        """
        return card

    def _estimate_suit(self, card):
        """
        Using the card feature determine suit.

        If we find something return the card, otherwise
        throw.
        """
        return card

    def _is_face_card(self, card):
        """
        Determine if we have a face card
        return any updates to the card and the state.
        """
        return False, card

    def _estimate_face_card(self, card):
        """
        Determine if we have a face card K/Q/J and some A
        """
        return card

    def _estimate_rank(self, card):
        """
        Determine the rank and reutrn the card otherwise throw.
        """
        return card

    def _is_non_standard_card(self, card):
        """
        Determine if our card is not a normal card like a joker
        the backside of a card or something similar. Return either the
        card or throw.
        """
        return card

    def _refine_estimates(self, card):
        """
        Do a post process step if we want.
        e.g. find corners with sub-pix accuracy to
        do a swell overlay. Do any numpy sanitation
        for Seer.
        """
        return card
