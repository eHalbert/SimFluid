/* SimShip by Edouard Halbert
This work is licensed under a Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License
http://creativecommons.org/licenses/by-nc-nd/4.0/ */

#include "Sky.h"

extern uint32_t g_FramesInFlight;

// Extracted table, format (temperature, r, g, b) from 1000K to 10000K, 500K/step
static const struct { float k, r, g, b; } blackbodyLUT[] = {
    {1000, 1.000f, 0.036f, 0.000f},
    {1500, 1.000f, 0.252f, 0.033f},
    {2000, 1.000f, 0.396f, 0.127f},
    {2500, 1.000f, 0.522f, 0.210f},
    {3000, 1.000f, 0.641f, 0.326f},
    {3500, 1.000f, 0.739f, 0.442f},
    {4000, 1.000f, 0.812f, 0.549f},
    {4500, 1.000f, 0.873f, 0.647f},
    {5000, 1.000f, 0.921f, 0.724f},
    {5500, 1.000f, 0.959f, 0.817f},
    {6000, 1.000f, 0.990f, 0.906f},
    {6500, 1.000f, 1.000f, 1.000f},
    {7000, 0.910f, 0.968f, 1.000f},
    {8000, 0.780f, 0.913f, 1.000f},
    {9000, 0.669f, 0.862f, 1.000f},
    {10000,0.568f, 0.816f, 1.000f}
};
vec3 blackbody_to_rgb(float tempK)
{
    // Clamp at the bounds of the table:
    if (tempK <= blackbodyLUT[0].k) return vec3(blackbodyLUT[0].r, blackbodyLUT[0].g, blackbodyLUT[0].b);
    if (tempK >= blackbodyLUT[15].k) return vec3(blackbodyLUT[15].r, blackbodyLUT[15].g, blackbodyLUT[15].b);

    // Linear interpolation
    for (int i = 0; i < 15; ++i)
    {
        if (tempK >= blackbodyLUT[i].k && tempK < blackbodyLUT[i + 1].k)
        {
            float t = (tempK - blackbodyLUT[i].k) / (blackbodyLUT[i + 1].k - blackbodyLUT[i].k);
            vec3 rgb0(blackbodyLUT[i].r, blackbodyLUT[i].g, blackbodyLUT[i].b);
            vec3 rgb1(blackbodyLUT[i + 1].r, blackbodyLUT[i + 1].g, blackbodyLUT[i + 1].b);
            return glm::mix(rgb0, rgb1, t);
        }
    }
    return vec3(1.0, 1.0, 1.0); // fallback
}


Sky::Sky(vec2 pos, int width, int height)
{
    Longitude = pos.x;
    Latitude = pos.y;
    mWidth = width;
    mHeight = height;

    SetNow();
}
Sky::~Sky() {}
void Sky::SetObserver(vec2 pos)
{
    Longitude = pos.x;
    Latitude = pos.y;
}

void Sky::SetNow()
{
    // Get the current time in UTC
    chrono::system_clock::time_point now = chrono::system_clock::now();
    time_t now_c = chrono::system_clock::to_time_t(now);

    // Convert to tm structure in UTC
    tm tm_utc;
    gmtime_s(&tm_utc, &now_c);

    // Create the ln_date structure with UTC values
    struct ln_date date;
    date.years = tm_utc.tm_year + 1900;
    date.months = tm_utc.tm_mon + 1;
    date.days = tm_utc.tm_mday;
    date.hours = tm_utc.tm_hour;
    date.minutes = tm_utc.tm_min;
    date.seconds = 0.0;

    SunHour = date.hours;
    SunMinute = date.minutes;
    tmTimeStored = tm_utc;

    UpdateData(date);
}
sHM Sky::GetNow()
{
    // Get the current time in UTC
    chrono::system_clock::time_point now = chrono::system_clock::now();
    time_t now_c = chrono::system_clock::to_time_t(now);

    // Convert to tm structure in UTC
    tm tm_utc;
    gmtime_s(&tm_utc, &now_c);

    sHM hm{};
    hm.hour = SunHour = tm_utc.tm_hour;
    hm.minute = SunMinute = tm_utc.tm_min;

    return hm;
}
void Sky::SetTime(int hour, int minute)
{
    // Get the current time in UTC
    chrono::system_clock::time_point now = chrono::system_clock::now();
    time_t now_c = chrono::system_clock::to_time_t(now);

    // Convert to tm structure in UTC
    tm tm_utc;
    gmtime_s(&tm_utc, &now_c);

    // Create the ln_date structure with UTC values
    struct ln_date date;
    date.years = tm_utc.tm_year + 1900;
    date.months = tm_utc.tm_mon + 1;
    date.days = tm_utc.tm_mday;
    date.hours = hour;
    date.minutes = minute;
    date.seconds = 0.0;

    SunHour = hour;
    SunMinute = minute;
    tmTimeStored = tm_utc;

    UpdateData(date);
}
sHM Sky::GetTime()
{
    // Get the current time in UTC
    chrono::system_clock::time_point now = chrono::system_clock::now();
    time_t now_c = chrono::system_clock::to_time_t(now);

    // Convert to tm structure in UTC
    tm tm_utc;
    gmtime_s(&tm_utc, &now_c);

    // Retrieve timezone offset in seconds
    long timezone_seconds = 0;
    int daylight_savings = 0;
    
    _tzset(); // initialize the internal timezone variable
    _get_timezone(&timezone_seconds); // get offset in seconds
    _get_daylight(&daylight_savings);  // Check if the timezone has a DST rule

    // Check if DST is currently active via tm_isdst
    tm tm_local;
    localtime_s(&tm_local, &now_c);

    int dst_offset = (daylight_savings && tm_local.tm_isdst > 0) ? 3600 : 0;

    // Conversion to rounded hours, subtracting the active DST
    int timezone_hours = static_cast<int>((-timezone_seconds + dst_offset) / 3600);

    int hour = tm_utc.tm_hour - tmTimeStored.tm_hour;
    int minute = tm_utc.tm_min - tmTimeStored.tm_min;
    if (SunHour + hour > 23)
        hour = 0;
    if (SunMinute + minute > 59)
    {
        if (SunHour + hour < 23)
        {
            SunMinute = 0;
            SunHour++;
        }
        else
        {
            SunMinute = 59;
            SunHour = 23;
        }
    }
    sHM hm{};
    hm.hour = glm::clamp(SunHour + hour, 0, 23);
    hm.minute = glm::clamp(SunMinute + minute, 0, 59);
    hm.timezoneOffsetHours = timezone_hours;

    return hm;
}
void Sky::UpdateData(ln_date& date)
{
#pragma region Sun and Moon
    // Calculate the Julian day
    double JD = ln_get_julian_day(&date);

    struct ln_lnlat_posn observer;
    observer.lng = Longitude;
    observer.lat = Latitude;

    // Sun
    struct ln_equ_posn sun;
    ln_get_solar_equ_coords(JD, &sun);

    struct ln_hrz_posn position;
    ln_get_hrz_from_equ(&sun, &observer, JD, &position);

    SunElevation = 90.0f - position.alt;
    SunAzimut = position.az + 180.0f;

    while (SunElevation < 0.0f)     SunElevation += 360.0f;
    while (SunElevation > 360.0f)   SunElevation -= 360.0f;
    while (SunAzimut < 0.0f)        SunAzimut += 360.0f;
    while (SunAzimut > 360.0f)      SunAzimut -= 360.0f;

    float radPhi = glm::radians(SunAzimut + 270.0f);
    float radTheta = glm::radians(SunElevation);
    SunDirection = vec3(sin(radTheta) * cos(radPhi), cos(radTheta), sin(radTheta) * sin(radPhi));
    SunPosition = SunDirection * SunDistance;
#pragma endregion

#pragma region Exposure
    const float minExposure = 0.02f;
    const float maxExposure = 1.0f;
    const float maxNightAngle = 108.0f;      // beginning of the "dark night"
    const float plateauStart = 65.0f;        // start of decay (adjusts according to desired effect)

    // Clamp the angle
    float clampedAngle = SunElevation;
    if (clampedAngle < 0.0f) clampedAngle = 0.0f;
    if (clampedAngle > maxNightAngle) clampedAngle = maxNightAngle;

    // Plateau at 1.0 on the sunny edge
    if (clampedAngle <= plateauStart)
    {
        Exposure = maxExposure;
        return;
    }

    // Smoothstep transition (Hermite/Cubic)
    // x = 0 when we start to decrease; x = 1 when we are in the night
    float t = (clampedAngle - plateauStart) / (maxNightAngle - plateauStart);
    // Smoothstep : 3t² - 2t³
    float smooth = 3.0f * t * t - 2.0f * t * t * t;
    Exposure = (1.0f - smooth) * maxExposure + smooth * minExposure;
#pragma endregion

#pragma region Color of the sun
    // Empirical model: visible sun temperature as a function of elevation, 2000K (red horizon) to 6500K (white zenith)
    float t_min = 2000.0f, t_max = 6500.0f;

    float tt = glm::clamp(glm::radians(SunElevation) / 1.5708f, 0.0f, 1.0f);
    tt = 1.0f - tt;  // t=1 (zenith), t=0 (horizon)
    float tempK = glm::mix(t_min, t_max, tt);

    vec3 temp = blackbody_to_rgb(tempK);
    SunEmissive = temp;
    SunDiffuse = temp;
    SunSpecular = temp;
#pragma endregion

#pragma region Pct of sun elevation
    time_t now_c;
    ln_get_timet_from_julian(JD, &now_c);
    mAlmanac = Nova::getSunMoonAlmanac(now_c, Latitude, Longitude);

    double sunRise = static_cast<double>(mAlmanac.sunRise % 86400) / 3600.0;
    double sunTransit = static_cast<double>(mAlmanac.sunTransit % 86400) / 3600.0;
    double sunSet = static_cast<double>(mAlmanac.sunSet % 86400) / 3600.0;
    double currentHour = static_cast<double>(now_c % 86400) / 3600.0f;

    double pct = 0.0;

    // Between sunrise and transit: 0 → 1
    if (currentHour >= sunRise && currentHour <= sunTransit)
        pct = (currentHour - sunRise) / (sunTransit - sunRise);
    // Between transit and sunset: 1 → 0  
    else if (currentHour > sunTransit && currentHour <= sunSet)
        pct = 1.0 - (currentHour - sunTransit) / (sunSet - sunTransit);
    // Before sunrise or after sunset: 0

    SunPctElevation = std::clamp(static_cast<float>(pct), 0.0f, 1.0f);
#pragma endregion
}


