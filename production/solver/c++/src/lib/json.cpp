/**
 * JSON OUTPUT (./c++/lib/json.cpp)
 * JSON ouput class
 */

#include <iostream>
#include <string>
#include <vector>
#include <sstream>
using namespace std;

class JSON {
 public:
   JSON();
   ~JSON();
   void append(string const& name, vector<float> const& vec);
   void append(string const& name, vector<double> const& vec);
   void append(string const& name, vector<long int> const& vec);
   void append(string const& name, vector<int> const& vec);
   void append(string const& name, double const& scalar);
   void append(string const& name, float const& scalar);
   void append(string const& name, int const& scalar);
   void append(string const& name, long int const& scalar);

   string dump();
 private:
   string output;
};

JSON::JSON() {};
JSON::~JSON() {};

void JSON::append(string const& name, vector<float> const& vec) {

   if (this->output.size() > 1) {
      this->output += ",";
   }

   if (vec.size() == 1) {
      this->output += "\"" + name + "\":" + to_string(vec[0]);

   } else {
      this->output += "\"" + name + "\":[";

      for (int i = 0; i < vec.size(); i++) {
         if (i != 0) {
            this->output += ",";
         }

         this->output += to_string(vec[i]);
      }

      this->output += "]";
   }
};


void JSON::append(string const& name, vector<double> const& vec) {

   if (this->output.size() > 1) {
      this->output += ",";
   }

   if (vec.size() == 1) {
      this->output += "\"" + name + "\":" + to_string(vec[0]);

   } else {
      this->output += "\"" + name + "\":[";

      for (int i = 0; i < vec.size(); i++) {
         if (i != 0) {
            this->output += ",";
         }

         this->output += to_string(vec[i]);
      }

      this->output += "]";
   }
};

void JSON::append(string const& name, vector<int> const& vec) {

   if (this->output.size() > 1) {
      this->output += ",";
   }

   if (vec.size() == 1) {
      this->output += "\"" + name + "\":" + to_string(vec[0]);

   } else {
      this->output += "\"" + name + "\":[";

      for (int i = 0; i < vec.size(); i++) {
         if (i != 0) {
            this->output += ",";
         }

         this->output += to_string(vec[i]);
      }

      this->output += "]";
   }
};

void JSON::append(string const& name, vector<long int> const& vec) {

   if (this->output.size() > 1) {
      this->output += ",";
   }

   if (vec.size() == 1) {
      this->output += "\"" + name + "\":" + to_string(vec[0]);

   } else {
      this->output += "\"" + name + "\":[";

      for (int i = 0; i < vec.size(); i++) {
         if (i != 0) {
            this->output += ",";
         }

         this->output += to_string(vec[i]);
      }

      this->output += "]";
   }
};


void JSON::append(string const& name, float const& scalar) {
   if (this->output.size() > 1) {
      this->output += ",";
   }

   this->output += "\"" + name + "\":" + to_string(scalar);
};

void JSON::append(string const& name, double const& scalar) {
   if (this->output.size() > 1) {
      this->output += ",";
   }

   this->output += "\"" + name + "\":" + to_string(scalar);
};

void JSON::append(string const& name, int const& scalar) {
   if (this->output.size() > 1) {
      this->output += ",";
   }

   this->output += "\"" + name + "\":" + to_string(scalar);
};

void JSON::append(string const& name, long int const& scalar) {
   if (this->output.size() > 1) {
      this->output += ",";
   }

   this->output += "\"" + name + "\":" + to_string(scalar);
};


string JSON::dump() {
   return "{" + this->output + "}";
};