#include <iostream>
#include "manager.h"

int main() {
	Manager manager;
	std::string text = "Hello, my name is llama and I am a friendly assistant running on Aryan's GPU. My favorite flavor of ice cream is";
	std::cout << "Calling handle_request on: " << text << std::endl;
	std::optional<std::string> response = manager.handle_request(text);
	if (response.has_value()) {
		std::cout << "Response: " << response.value() << std::endl;
	} else {
		std::cout << "No response" << std::endl;
	}
}
