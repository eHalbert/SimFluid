/* SimShip by Edouard Halbert
This work is licensed under a Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License
http://creativecommons.org/licenses/by-nc-nd/4.0/ */

#pragma once

// 1. PROJECT
#include "Structures.h"
#include "Camera.h"
#include "Utility.h"
#include "vulkan_device.hpp"
#include "vulkan_swapchain.hpp"
#include "vulkan_texture.hpp"

// 2. LIB
#include <vulkan/vulkan.h>

// glm
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
using namespace glm;

extern "C"
{
#include "nova/nova.h"
}
#ifdef _DEBUG
#pragma comment(lib, "nova/Debug/nova.lib")
#else
#pragma comment(lib, "nova/Release/nova.lib")
#endif

// 3. WIN
#define NOMINMAX
#include <complex>
#include <vector>
using namespace std;


struct sHM
{
	int hour;
	int minute;
	int timezoneOffsetHours;
};


class Sky
{
public:
	Sky(vec2 pos, int width, int height);
	~Sky();

	void	SetObserver(vec2 pos);

	void	SetNow();
	sHM		GetNow();
	void	SetTime(int hour, int minute);
	sHM		GetTime();


	float	SunAzimut			= 0.0f;					// 0 = z-, 180 = z+ (anticlockwise)
	float	SunElevation		= 30.0f;				// 0 = y+, 180 = y- (clockwise)
	float	SunPctElevation		= 0.0f;					// % of sun elevation (0 = sun rise and sunset, 1 = sun at zenith)
	float	SunDistance			= 20000.0f;

	int		SunHour				= 12;					// Store the hours
	int		SunMinute			= 0;					// Store the minutes
	tm		tmTimeStored;

	float	Longitude			= -2.94097114f;
	float	Latitude			= 47.38162231f;

	vec3	SunPosition			= vec3(0.0f, 0.0f, 0.0f);
	vec3	SunDirection		= vec3(0.0f, 0.0f, 0.0f);

	vec3	SunEmissive			= vec3(1.0f, 0.98f, 0.92f);
	vec3	SunAmbient			= vec3(0.75f, 0.75f, 0.75f);
	vec3	SunDiffuse			= vec3(1.0f, 0.98f, 0.92f);
	vec3	SunSpecular			= vec3(1.0f, 0.98f, 0.92f);

	float	MistDensity			= 0.00005f;
	bool    bVisible			= true;

	float	Exposure			= 1.0f;

private:
	void	UpdateData(ln_date& date);

	int		mWidth, mHeight;
	Almanac	mAlmanac;
};
