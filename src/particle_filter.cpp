/*
 * particle_filter.cpp
 *
 *  Created on: November 24, 2018
 *      Author: Aravind Pillarisetti
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  // Set the number of particles.
  num_particles = 100;
  
  std::default_random_engine gen;
  std::normal_distribution<double> dist_x(x, std[0]);
  std::normal_distribution<double> dist_y(y, std[1]);
  std::normal_distribution<double> dist_t(theta, std[2]);

  // Create particles
  for(int idx=0; idx < num_particles; idx++) {
    Particle particle;

    //Initialize particle to first position (based on estimates of 
    //x, y, theta and their uncertainties from GPS) and all weights to 1. 
    particle.id     = idx;
    particle.weight = 1.0;

    // Add random Gaussian noise to each particle.
    particle.x = dist_x(gen);
    particle.y = dist_y(gen);
    particle.theta = dist_t(gen);

    particles.push_back(particle);
  }

  // initialize complete
  is_initialized = true;
  
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {

  // Normal distributions for sensor noise
  std::default_random_engine gen;
  std::normal_distribution<double> dist_x(0, std_pos[0]);
  std::normal_distribution<double> dist_y(0, std_pos[1]);
  std::normal_distribution<double> dist_t(0, std_pos[2]);

  // Add measurements to each particle and add random Gaussian noise
  for(auto&& particle : particles) {

    double th = particle.theta;
    if (fabs(yaw_rate) < 0.00001) {
      // avoid divide by zero
      particle.x += velocity * delta_t * cos(th);
      particle.y += velocity * delta_t * sin(th);
      
    } else {
      particle.x += (velocity/yaw_rate) * (sin(th + yaw_rate*delta_t) - sin(th));
      particle.y += (velocity/yaw_rate) * (cos(th) - cos(th + yaw_rate*delta_t));
      particle.theta += yaw_rate * delta_t;
    }

    particle.x += dist_x(gen);
    particle.y += dist_y(gen);
    particle.theta += dist_t(gen);

  }

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
  // Find the predicted measurement that is closest to each observed measurement and assign the 
  // observed measurement to this particular landmark.
  // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
  //   implement this method and use it as a helper during the updateWeights phase.

  for (auto&& obs : observations) {

    // initialize min distance
    double minDist = numeric_limits<double>::max();
    
    int map_id = -1;
    
    for (const auto& pred : predicted) {
	
      double cur_dist = pow( (obs.x - pred.x), 2) + pow( (obs.y- pred.y), 2);
      
      if (cur_dist < minDist) {
	minDist = cur_dist;
	map_id = pred.id;
      }
    }
    
    obs.id = map_id;
  }
}


void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
  // Update the weights of each particle using a mult-variate Gaussian distribution. You can read
  //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
  // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
  //   according to the MAP'S coordinate system. You will need to transform between the two systems.
  //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
  //   The following is a good resource for the theory:
  //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
  //   and the following is a good resource for the actual equation to implement (look at equation 
  //   3.33
  //   http://planning.cs.uiuc.edu/node99.html

  double sx = std_landmark[0];
  double sy = std_landmark[1];
  double factor = 1/(2*M_PI*sx*sy);
  double den1   = 2*pow(sx, 2);
  double den2   = 2*pow(sy,2);
  unsigned int lmsize = map_landmarks.landmark_list.size();
        
  //for each particle
  for (auto&& particle : particles) { 

    double px = particle.x;
    double py = particle.y;
    double pt = particle.theta;

    // Vector of landmark predictions
    vector<LandmarkObs> predictions;
    // for each landmark in the map
    for (unsigned int lIdx=0; lIdx < lmsize; lIdx++) {
      float lmx = map_landmarks.landmark_list[lIdx].x_f;
      float lmy = map_landmarks.landmark_list[lIdx].y_f;
      int lmId  = map_landmarks.landmark_list[lIdx].id_i;

      // consider landmarks within sensor range
      if( pow(lmx-px, 2) + pow(lmy-py, 2) <= pow(sensor_range,2) ) {

	// add predictions
	predictions.push_back(LandmarkObs{ lmId, lmx, lmy});
      }
    }

    // create list of observations from vehicle to map co-ordinates
    vector<LandmarkObs> transformed_obs;
    for (const auto& obs : observations) { 
      double tx = cos(pt)*obs.x - sin(pt)*obs.y + px;
      double ty = sin(pt)*obs.x + cos(pt)*obs.y + py;
      transformed_obs.push_back(LandmarkObs {obs.id, tx, ty});
    }

    // do dataAssociation for prediction to transformed
    dataAssociation(predictions, transformed_obs);

    // reinitialize weight
    particle.weight = 1.0;

    for (const auto& tobs : transformed_obs) {
      double ox = tobs.x;
      double oy = tobs.y;

      // get predicted x and y associated with current observation
      double predicted_x; double predicted_y;
      for (unsigned int pIdx=0; pIdx < predictions.size(); pIdx++) {
	if (predictions[pIdx].id  == tobs.id) {
	  predicted_x = predictions[pIdx].x;
	  predicted_y = predictions[pIdx].y;
	  break;
	}
      }

      // calculate weight of this observation
      double sw = ( factor*
		    exp( -( (pow(predicted_x-ox,2)/den1) + (pow(predicted_y-oy,2)/den2) ))
		    );

      // Multiply by weight of this observation
      if (sw == 0) {
	particle.weight *= 0.00001;
      } else {
	particle.weight *= sw;
      }
    }
    
  }
  
}

void ParticleFilter::resample() {
  // Resample particles with replacement with probability proportional to their weight. 
  // NOTE: You may find std::discrete_distribution helpful here.
  //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

  std::random_device rd;
  std::default_random_engine gen(rd());

  vector<Particle> resampleParticles;
  vector<double> weights;

  // get all current weights
  for(const auto& particle : particles) { 
    weights.push_back(particle.weight); 
  }

  //use discrete distribution to return particles by weight
  for(int idx=0; idx < num_particles; idx++) {
    discrete_distribution<int> index(weights.begin(), weights.end());
    resampleParticles.push_back(particles[index(gen)]);
  }

  particles = resampleParticles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
  vector<int> v = best.associations;
  stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseX(Particle best)
{
  vector<double> v = best.sense_x;
  stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseY(Particle best)
{
  vector<double> v = best.sense_y;
  stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
