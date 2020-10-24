import * as React from "react";
import {
  faAngry,
  faFrown,
  faMeh,
  faSmileBeam,
  faGrinStars,
  faEllipsisV,
} from "@fortawesome/free-solid-svg-icons";
import { get as _get } from "lodash";
import { IconProp } from "@fortawesome/fontawesome-svg-core";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import RecommendationControl from "./RecommendationControl";

import { getArtistLink, getTrackLink } from "../utils";
import Card from "../components/Card";
import APIService from "../APIService";

export type RecommendationCardProps = {
  apiUrl: string;
  recommendation: Recommendation;
  playRecommendation: (recommendation: Recommendation) => void;
  currentFeedback: RecommendationFeedBack | null;
  updateFeedback: (
    recordingMbid: string,
    rating: RecommendationFeedBack | null
  ) => void;
  className?: string;
  isCurrentUser: Boolean;
  currentUser?: ListenBrainzUser;
  newAlert: (
    alertType: AlertType,
    title: string,
    message: string | JSX.Element
  ) => void;
};

type RecommendationCardState = {
  feedback: RecommendationFeedBack | null;
};

export default class RecommendationCard extends React.Component<
  RecommendationCardProps,
  RecommendationCardState
> {
  APIService: APIService;
  playRecommendation: (recommendation: Recommendation) => void;

  constructor(props: RecommendationCardProps) {
    super(props);

    this.state = {
      feedback: props.currentFeedback || null,
    };

    this.APIService = new APIService(
      props.apiUrl || `${window.location.origin}/1`
    );

    this.playRecommendation = props.playRecommendation.bind(
      this,
      props.recommendation
    );
  }

  componentDidUpdate(prevProps: RecommendationCardProps) {
    const { currentFeedback } = this.props;
    if (currentFeedback !== prevProps.currentFeedback) {
      this.setState({ feedback: currentFeedback });
    }
  }

  submitFeedback = async (
    event: React.SyntheticEvent,
    rating: RecommendationFeedBack
  ) => {
    const {
      recommendation,
      currentUser,
      isCurrentUser,
      updateFeedback,
    } = this.props;
    /* Stop click propagation so that the Bootstrap dropdown does not close on feedback click */
    event.stopPropagation();
    if (isCurrentUser && currentUser?.auth_token) {
      const recordingMBID = _get(
        recommendation,
        "track_metadata.additional_info.recording_mbid"
      );
      try {
        const status = await this.APIService.submitRecommendationFeedback(
          currentUser.auth_token,
          recordingMBID,
          rating
        );
        if (status === 200) {
          this.setState({ feedback: rating });
          updateFeedback(recordingMBID, rating);
        }
      } catch (error) {
        this.handleError(
          `Error while submitting recommendation feedback - ${error.message}`
        );
      }
    }
  };

  deleteFeedback = async (event: React.SyntheticEvent) => {
    const {
      recommendation,
      currentUser,
      isCurrentUser,
      updateFeedback,
    } = this.props;
    /* Stop click propagation so that the Bootstrap dropdown does not close on feedback click */
    event.stopPropagation();
    if (isCurrentUser && currentUser?.auth_token) {
      const recordingMBID = _get(
        recommendation,
        "track_metadata.additional_info.recording_mbid"
      );
      try {
        const status = await this.APIService.deleteRecommendationFeedback(
          currentUser.auth_token,
          recordingMBID
        );
        if (status === 200) {
          this.setState({ feedback: null });
          updateFeedback(recordingMBID, null);
        }
      } catch (error) {
        this.handleError(
          `Error while deleting recommendation feedback - ${error.message}`
        );
      }
    }
  };

  handleError = (error: string | Error, title?: string): void => {
    const { newAlert } = this.props;
    if (!error) {
      return;
    }
    newAlert(
      "danger",
      title || "Error",
      typeof error === "object" ? error.message : error
    );
  };

  render() {
    const { recommendation, className } = this.props;
    const { feedback } = this.state;

    return (
      <Card
        onDoubleClick={this.playRecommendation}
        className={`recommendation-card row ${className}`}
      >
        <div className="col-xs-9">
          <div className="col-xs-11">
            <div className="track-details">
              <p title={recommendation.track_metadata.track_name}>
                {getTrackLink(recommendation)}
              </p>
              <p>
                <small
                  className="text-muted"
                  title={recommendation.track_metadata.artist_name}
                >
                  {getArtistLink(recommendation)}
                </small>
              </p>
            </div>
          </div>
        </div>
        <div className="col-xs-1 text-center">
          <div className="recommendation-controls">
            <>
              <FontAwesomeIcon
                icon={faEllipsisV as IconProp}
                className="dropdown-toggle"
                id="recommendationControlsDropdown"
                data-toggle="dropdown"
                aria-haspopup="true"
                aria-expanded="true"
              />
              <ul
                className="dropdown-menu dropdown-menu-right"
                aria-labelledby="recommendationControlsDropdown"
              >
                <RecommendationControl
                  icon={faAngry}
                  title="I never want to hear this again!"
                  action={(event: React.SyntheticEvent) =>
                    feedback === "hate"
                      ? this.deleteFeedback(event)
                      : this.submitFeedback(event, "hate")
                  }
                  className={`${feedback === "hate" ? " angry" : ""}`}
                />
                <RecommendationControl
                  icon={faFrown}
                  title="I don't like this!"
                  action={(event: React.SyntheticEvent) =>
                    feedback === "dislike"
                      ? this.deleteFeedback(event)
                      : this.submitFeedback(event, "dislike")
                  }
                  className={`${feedback === "dislike" ? " frown" : ""}`}
                />
                <RecommendationControl
                  icon={faMeh}
                  title="This is a bad recommendation!"
                  action={(event: React.SyntheticEvent) =>
                    feedback === "bad_recommendation"
                      ? this.deleteFeedback(event)
                      : this.submitFeedback(event, "bad_recommendation")
                  }
                  className={`${
                    feedback === "bad_recommendation" ? " meh" : ""
                  }`}
                />
                <RecommendationControl
                  icon={faSmileBeam}
                  title="I like this!"
                  action={(event: React.SyntheticEvent) =>
                    feedback === "like"
                      ? this.deleteFeedback(event)
                      : this.submitFeedback(event, "like")
                  }
                  className={`${feedback === "like" ? " smile" : ""}`}
                />
                <RecommendationControl
                  icon={faGrinStars}
                  title="I really love this!"
                  action={(event: React.SyntheticEvent) =>
                    feedback === "love"
                      ? this.deleteFeedback(event)
                      : this.submitFeedback(event, "love")
                  }
                  className={`${feedback === "love" ? " grin" : ""}`}
                />
              </ul>
            </>
          </div>
        </div>
      </Card>
    );
  }
}
