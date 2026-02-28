#include <iostream>
#include "manager.h"

int main() {
	Manager manager;
	std::string text = "Hello, world!";
	std::cout << "Calling handle_request on: " << text << std::endl;
	std::optional<std::string> response = manager.handle_request(text);
	if (response.has_value()) {
		std::cout << "Response: " << response.value() << std::endl;
	} else {
		std::cout << "No response" << std::endl;
	}
}
