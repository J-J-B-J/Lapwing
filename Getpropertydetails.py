"""Given the HTML source code for a realestate.com.au page, get the details of
the property."""
from bs4 import BeautifulSoup
from realestate_com_au import RealestateComAu
import numpy as np
import matplotlib.pyplot as plt
import requests


def get_properties_in_greenhills():
    """Get a list of properties in Greenhills Beach as Listing objects"""
    api = RealestateComAu()
    listings = api.search(
        locations=["Greenhills Beach, NSW 2230"],
        channel="sold"
    )
    new_listings = []
    for listing in listings:
        if listing.suburb == "Greenhills Beach":
            new_listings.append(listing)
    listings = new_listings
    del new_listings
    return listings


def get_property_bed_nums(listings: [any]):
    """Plot the number of bedrooms for a list of properties"""
    fig, ax = plt.subplots()

    bedrooms = [listing.bedrooms if type(listing.bedrooms) == int else 0 for
                listing in listings]
    bedrooms_counts = {}
    for i in range(1, max(bedrooms)+1):
        bedrooms_counts[i] = bedrooms.count(i)

    ax.plot(
        bedrooms_counts.keys(),
        bedrooms_counts.values(),
        marker="x"
    )
    ax.set_title("Bedroom Numbers for houses in Greenhills Beach, NSW 2230")
    ax.set_xlabel("Number of bedrooms")
    ax.set_ylabel("Number of houses")
    plt.show()


def get_property_bath_nums(listings: [any]):
    """Plot the number of bathrooms for a list of properties"""
    fig, ax = plt.subplots()

    bathrooms = [listing.bathrooms if type(listing.bathrooms) == int else 0 for
                 listing in listings]
    bathrooms_counts = {}
    for i in range(1, max(bathrooms) + 1):
        bathrooms_counts[i] = bathrooms.count(i)

    ax.plot(
        bathrooms_counts.keys(),
        bathrooms_counts.values(),
        marker="x"
    )
    ax.set_title("Bathroom Numbers for houses in Greenhills Beach, NSW 2230")
    ax.set_xlabel("Number of bathrooms")
    ax.set_ylabel("Number of houses")
    plt.show()


def get_property_land_size(listings: [any]):
    """Plot the number of bathrooms for a list of properties"""
    fig, ax = plt.subplots()

    new_listings = []
    for listing in listings:
        if listing.building_size and listing.land_size:
            new_listings.append(listing)
    listings = new_listings
    del new_listings

    ax.scatter(
        [listing.land_size for listing in listings],
        [listing.building_size for listing in listings],
        marker="x"
    )
    ax.set_title("Land vs Building sizes for houses in Greenhills Beach, NSW "
                 "2230")
    ax.set_xlabel("Land Size")
    ax.set_ylabel("Building Size")
    plt.show()


def get_property_parking_nums(listings: [any]):
    """Plot the number of parking spaces for a list of properties"""
    fig, ax = plt.subplots()

    parking_spaces = [
        listing.parking_spaces if type(listing.parking_spaces) == int
        else 0 for listing in listings
    ]
    parking_spaces_counts = {}
    for i in range(1, max(parking_spaces) + 1):
        parking_spaces_counts[i] = parking_spaces.count(i)

    ax.plot(
        parking_spaces_counts.keys(),
        parking_spaces_counts.values(),
        marker="x"
    )
    ax.set_title("Parking Space Numbers for houses in Greenhills Beach, NSW "
                 "2230")
    ax.set_xlabel("Number of Parking Spaces")
    ax.set_ylabel("Number of houses")
    plt.show()


def save_property_floorplans(listings: [any]):
    """Save all the floorplans for a list of listings"""
    i = 0
    for listing in listings:
        if not listing.images_floorplans:
            print(f"Listing has no floorplans: {listing.full_address}")
            continue
        for image in listing.images_floorplans:
            try:
                with open(f"img/floorplans/{i}.{image.link.split('.')[-1]}",
                          "wb") as f:
                    f.write(requests.get(image.link).content)
            except:
                print(f" Error saving image (?): {image.link}")
                continue
            else:
                i += 1


def save_property_images(listings: [any]):
    """Save all the images for a list of listings"""
    i = 0
    for listing in listings:
        if not listing.images:
            print(f"Listing has no images: {listing.full_address}")
            continue
        for image in listing.images:
            try:
                with open(f"img/other/{i}.{image.link.split('.')[-1]}",
                          "wb") as f:
                    f.write(requests.get(image.link).content)
            except:
                print(f" Error saving image (?): {image.link}")
                continue
            else:
                i += 1


if __name__ == "__main__":
    print(get_properties_in_greenhills())
